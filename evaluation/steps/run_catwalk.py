import logging
import time
from pydoc import locate
from typing import Any, Dict, List, Optional, Sequence

from catwalk.dependencies.lm_eval.utils import simple_parse_args_string
from catwalk.model import Model
from catwalk.models import MODELS, add_decoder_only_model
from catwalk.task import rc_metrics
from catwalk.tasks import TASKS, get_instances
from catwalk.tasks.tasks_lm import TASKS_LM
from catwalk.utils import guess_instance_id
from tango.step import Step

logger = logging.getLogger(__name__)


@Step.register("construct-task")
class ConstructTaskDict(Step):
    VERSION = "003"

    def run(self, task_name: str) -> Dict:  # Task:
        # TODO: deal with task_file case, it's the reason for task_dict.
        task_dict = {"name": task_name}
        try:
            task_obj = TASKS_LM.get(task_name, TASKS.get(task_name))
        except KeyError:
            raise KeyError(f"{task_name} not found")

        task_dict["task_obj"] = task_obj

        task_dict = self._update_task_metrics(task_dict)
        task_dict = self._update_unconditioned_prompt(task_dict)
        return task_dict

    @classmethod
    def _update_task_metrics(cls, task_dict: Dict) -> Dict:
        task_name = task_dict["name"]
        task_obj = task_dict["task_obj"]
        if "relative_improvement" in task_obj.metrics or "primary_metric" in task_dict:
            kwargs = {}
            if "primary_metric" in task_dict:
                kwargs["primary"] = task_dict["primary_metric"]
                logger.info(f"Overriding metric for {task_name} with rc_metrics ({kwargs})")
            else:
                logger.warning(f"Overriding 'acc' metric for {task_name} with rc_metrics")
            task_obj.metrics = {}
            task_obj.add_metrics(rc_metrics(**kwargs))
        task_dict["task_obj"] = task_obj
        return task_dict

    @classmethod
    def _update_unconditioned_prompt(cls, task_dict: Dict) -> Dict:
        task_name = task_dict["name"]
        task_obj = task_dict["task_obj"]
        if "unconditioned_prompt" not in task_dict:
            if hasattr(task_obj, "inner_task") and hasattr(task_obj.inner_task, "unconditioned_prompt"):
                prompt = task_obj.inner_task.unconditioned_prompt()
                logger.info(f"Using unconditioned prompt for {task_name}: '{prompt}'")
                task_dict["unconditioned_prompt"] = prompt
        return task_dict


@Step.register("construct-catwalk-model")
class ConstructCatwalkModel(Step):
    VERSION = "002"

    def run(self, model: str, model_path: Optional[str] = None, model_class: Optional[str] = None) -> Model:
        # TODO: ugly. clean up later.
        if model not in MODELS:
            prefix_split = model.split("::", 1)
            model_name = prefix_split[-1]
            # prefix = "" if len(prefix_split) == 1 else prefix_split[0]+"::"
            model_args = simple_parse_args_string(model_name)
            if "pretrained" not in model_args:
                raise ValueError(f"Unknown model {model}")
            hf_name = model_args["pretrained"]
            del model_args["pretrained"]
            if model_path:
                hf_name = model_path
            if model_class:
                model_args["model_class"] = locate(model_class)

                # TODO: why do we do this?
                # # Assuming tokenizer will be loaded with model, so fail if trying to load it otherwise
                # model_args['pretrained_tokenizer_name_or_path'] = 'UnknownTokenizer'
                model_args["pretrained_tokenizer_name_or_path"] = model_path

            add_decoder_only_model(model_name, hf_name, **model_args)
        return MODELS[model]


@Step.register("predict-and-calculate-metrics")
class PredictAndCalculateMetricsStep(Step):
    VERSION = "001"

    def run(
        self,
        model: Model,  # TODO: we need a catwalk version of the olmo model
        task_dict: Dict,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        random_subsample_seed: Optional[int] = None,
        model_max_length: int = 2048,
        max_batch_tokens: int = 20480,
        batch_size: int = 32,
        **kwargs,
    ) -> Dict:
        task_name = task_dict["name"]
        task = task_dict["task_obj"]

        start_time = time.time()

        instances = get_instances(task, split, limit, random_subsample_seed)
        predictions = [
            result
            for result in model.predict(
                task,
                instances,
                batch_size=batch_size,
                model_max_length=model_max_length,
                max_batch_tokens=max_batch_tokens,
                **kwargs,
            )
        ]
        metrics = model.calculate_metrics(task, predictions)  # this updates the `predictions` object too

        end_time = time.time()

        instance_predictions = self._instance_predictions_map_list(
            instances, predictions, task_dict.get("keep_instance_fields", None)
        )

        if instance_predictions:
            self.logger.info(f"First instance details for task {task_name}: {instance_predictions[0]}")

        output = {
            "task": task_dict["name"],
            "task_options": kwargs,  # model prediction kwargs
            "metrics": metrics,
            "num_instances": len(instances),
            "processing_time": end_time - start_time,
            "instance_predictions": instance_predictions,
        }

        # TODO: do we need to save to external file if tango is already saving it as step result? I think not.

        return output

    @classmethod
    def _instance_predictions_map_list(
        cls, instances, predictions, keep_instance_fields: Optional[List] = None
    ) -> List:
        instance_predictions = []

        for idx, (instance, pred) in enumerate(zip(instances, predictions)):
            instance_id = guess_instance_id(instance, idx=idx)  # dict

            if keep_instance_fields:
                for field in keep_instance_fields:
                    if field in instance:
                        instance_id[field] = instance[field]

            prediction = pred.get("prediction", pred)

            model_input = None
            # Move model_input from prediction if need be
            if "model_input" in pred:
                model_input = pred["model_input"]
                if "model_input" in prediction:
                    del prediction["model_input"]

            instance_pred = {"instance": instance_id, "prediction": prediction}
            if model_input is not None:
                instance_pred["model_input"] = model_input
            instance_predictions.append(instance_pred)

        return instance_predictions


@Step.register("post-process-outputs")
class PostProcessOutputPerTaskSpec(Step):
    # TODO: save as required csv instead.
    VERSION = "002"

    def run(self, model: str, outputs: List[Dict]) -> List:
        metrics_printed = []
        for d in outputs:
            metrics_printed.append(f" *** {d['task']} ***  (n = {d['num_instances']})  [{d['task_options']}]")
            metrics = {}
            # Code is a bit confused about nestedness of metrics
            for metric_name, metric in d["metrics"].items():
                if isinstance(metric, dict):
                    metrics.update(metric)
                else:
                    metrics[metric_name] = metric
            for metric_name, metric in metrics.items():
                metrics_printed.append(f"    {metric_name}: {metric}")
            metrics_printed.append("-----------------")
        logger.info("Overall metrics:\n  " + "\n".join(metrics_printed))
        return metrics_printed
