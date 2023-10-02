import copy
import logging
import os
import time
from datetime import datetime
from pydoc import locate
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz
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
    VERSION = "004"

    def run(self, task_name: str, task_rename: Optional[str] = None, **kwargs) -> Dict:  # Task:
        task_dict = {"name": task_name}
        try:
            task_obj = TASKS_LM.get(task_name, TASKS.get(task_name))
        except KeyError:
            raise KeyError(f"{task_name} not found")

        # TODO: not clean.
        if hasattr(task_obj, "clone") and "files" in kwargs:
            if "EVAL_DATA_PATH" in os.environ:
                files = [os.path.join(os.environ["EVAL_DATA_PATH"], filename) for filename in kwargs["files"]]
            else:
                files = kwargs["files"]
            task_obj = task_obj.clone(files=files)
        task_dict["task_obj"] = task_obj

        if task_rename:
            task_dict["name"] = task_rename

        task_dict.update(**kwargs)

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
    # CACHEABLE = False

    def run(
        self,
        model_path: str,
        model_class: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> Model:
        if "::" in model_path:
            model = model_path
        else:
            model = f"lm::pretrained={model_path.replace('/', '-')}"

        if model not in MODELS:
            prefix_split = model.split("::", 1)
            model_name = prefix_split[-1]
            # prefix = "" if len(prefix_split) == 1 else prefix_split[0]+"::"
            model_args = simple_parse_args_string(model_name)
            model_args.update({"revision": revision, "trust_remote_code": trust_remote_code})
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


DEFAULT_PREDICTION_KWARGS: Dict[str, Any] = {
    "model_max_length": 2048,
    "max_batch_tokens": 20480,
    "batch_size": 32,
    "limit": 1000,
    "split": "validation",
    "random_subsample_seed": 1234,
}


@Step.register("predict-and-calculate-metrics")
class PredictAndCalculateMetricsStep(Step):
    VERSION = "003"

    def run(
        self,
        model: Model,
        task_dict: Dict,
        split: Optional[str] = DEFAULT_PREDICTION_KWARGS["split"],
        limit: Optional[int] = DEFAULT_PREDICTION_KWARGS["limit"],
        random_subsample_seed: Optional[int] = DEFAULT_PREDICTION_KWARGS["random_subsample_seed"],
        model_max_length: int = DEFAULT_PREDICTION_KWARGS["model_max_length"],
        max_batch_tokens: int = DEFAULT_PREDICTION_KWARGS["max_batch_tokens"],
        batch_size: int = DEFAULT_PREDICTION_KWARGS["batch_size"],
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
                unconditioned_prompt=task_dict.get("unconditioned_prompt", None),
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

        task_options = {key: val for key, val in task_dict.items() if key not in ["name", "task_obj"]}
        output = {
            "task": task_dict["name"],
            "task_options": task_options,  # model prediction kwargs
            "metrics": metrics,
            "num_instances": len(instances),
            "processing_time": end_time - start_time,
            "instance_predictions": instance_predictions,
        }

        return output

    @classmethod
    def _instance_predictions_map_list(
        cls, instances, predictions, keep_instance_fields: Optional[List] = None
    ) -> List:
        instance_predictions = []

        for idx, (instance, pred) in enumerate(zip(instances, predictions)):
            instance_id = guess_instance_id(instance, idx=idx)  # dict

            if keep_instance_fields:
                for field in (instance if keep_instance_fields == "all" else keep_instance_fields):
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


@Step.register("write-outputs-as-rows")
class WriteOutputsAsRows(Step):
    VERSION = "001"

    def run(
        self, models: List[str], outputs: List[Dict], prediction_kwargs: List[Dict], gsheet: Optional[str] = None
    ) -> List:
        tsv_outputs = []
        for idx, d in enumerate(outputs):
            model = models[idx]
            pred_kwargs = copy.deepcopy(DEFAULT_PREDICTION_KWARGS)
            pred_kwargs.update(prediction_kwargs[idx])
            row = {}
            row["date"] = datetime.now(tz=pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            row["model"] = model
            row["full_model"] = f"lm::pretrained={model}"
            metrics_dict = list(d["metrics"].values())[0]

            # TODO: Very hacky.
            if "primary_metric" not in metrics_dict:
                primary_metric = "f1"
            else:
                primary_metric = metrics_dict["primary_metric"]

            row["task"] = d["task"]
            row["primary_metric"] = primary_metric
            row["metric"] = metrics_dict[primary_metric]
            row["processing_time"] = d["processing_time"]
            row["num_instances"] = d["num_instances"]
            row["tango_workspace"] = self.workspace.url
            row["tango_step"] = self.unique_id

            row.update(pred_kwargs)
            tsv_outputs.append(row)

        if gsheet:
            self._write_to_gsheet(gsheet, tsv_outputs)

        return tsv_outputs

    def _write_to_gsheet(self, gsheet: str, rows: List[Dict]):
        import pygsheets

        client = pygsheets.authorize(service_account_json=os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"])
        sheet = client.open(gsheet)
        worksheet = sheet[0]  # TODO: pass in sheet title, etc.
        current_df = worksheet.get_as_df()
        new_df = pd.concat([current_df, pd.DataFrame(rows)])
        worksheet.set_dataframe(new_df, (1, 1), nan="")
