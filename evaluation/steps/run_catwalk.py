import logging
from typing import Dict, Optional, Sequence, Any
from tango.step import Step, step
from catwalk.tasks import TASKS, Task
from catwalk.task import rc_metrics
from catwalk.model import Model
from catwalk.models import MODELS, add_decoder_only_model
from catwalk.tasks.tasks_lm import TASKS_LM
from catwalk.tasks import get_instances
from catwalk.dependencies.lm_eval.utils import simple_parse_args_string
from pydoc import locate

logger = logging.getLogger(__name__)


def update_task_metrics(task_dict: Dict) -> Dict:
    task_name = task_dict["name"]
    task_obj = task_dict["task_obj"]
    if "relative_improvement" in task_obj.metrics or 'primary_metric' in task_dict:
        kwargs = {}
        if 'primary_metric' in task_dict:
            kwargs['primary'] = task_dict['primary_metric']
            logger.info(f"Overriding metric for {task_name} with rc_metrics ({kwargs})")
        else:
            logger.warning(f"Overriding 'acc' metric for {task_name} with rc_metrics")
        task_obj.metrics = {}
        task_obj.add_metrics(rc_metrics(**kwargs))
    task_dict["task_obj"] = task_obj
    return task_dict


def update_unconditioned_prompt(task_dict: Dict) -> Dict:
    task_name = task_dict["name"]
    task_obj = task_dict["task_obj"]
    if 'unconditioned_prompt' not in task_dict:
        if hasattr(task_obj, "inner_task") and hasattr(task_obj.inner_task, "unconditioned_prompt"):
            prompt = task_obj.inner_task.unconditioned_prompt()
            logger.info(f"Using unconditioned prompt for {task_name}: '{prompt}'")
            task_dict['unconditioned_prompt'] = prompt
    return task_dict


@step("construct-task", version="001")
def construct_task_dict(task_name: str) -> Task:
    # TODO: deal with task_file case, it's the reason for task_dict.
    task_dict = {"name": task_name}
    try:
        task_obj = TASKS_LM.get(task_name, TASKS.get(task_name))
    except KeyError:
        raise KeyError(f"{task_name} not found")

    task_dict["task_obj"] = task_obj

    task_dict = update_task_metrics(task_dict)
    task_dict = update_unconditioned_prompt(task_dict)
    return task_dict["task_obj"]


@Step.register("simple-predict")
class SimplePredictStep(Step):
    def run(
        self,
        model: Model,  # TODO: we need a catwalk version of the olmo model
        task: Task,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        random_subsample_seed: Optional[int] = None,
        **kwargs
    ) -> Sequence[Any]:

        results = []
        instances = get_instances(task, split, limit, random_subsample_seed)
        for result in model.predict(task, instances, **kwargs):
            results.append(result)
        return results


@Step.register("simple-calculate-metrics")
class SimpleCalculateMetricsStep(Step):
    def run(
        self,
        model: Model,
        task: Task,
        predictions: Sequence[Any]
    ) -> Dict[str, float]:
        metrics = model.calculate_metrics(task, predictions)
        return metrics, predictions


def get_all_task_dicts(
    task: str,
    limit: int = 1000,
    num_shots: int = 0,
    num_recorded_inputs: int = 3,
    split: Optional[str] = None,
    batch_size: int = 32,
    fewshot_seed: Optional[int] = None,
    model_max_length: Optional[int] = None,
    max_batch_tokens: Optional[int] = None,
    random_subsample_seed: Optional[int] = None,
):
    # TODO: deal with task_file case separately (this is the case for )
    tasks = []
    task_list = task.split(" ")

    # TODO: deal with task_options
    # TODO: why do we need task_rename?
    for task_name in task_list:  # this split can happen in the config.
        task_dict = {"name": task_name}
        try:
            task_obj = TASKS_LM.get(task_name, TASKS.get(task_name))
        except KeyError:
            raise KeyError(f"{task_name} not found")
        task_dict["task_obj"] = task_obj
        task_dict["split"] = split or task_obj.default_split

        task_dict = update_task_metrics(task_dict)
        task_dict = update_unconditioned_prompt(task_dict)

        tasks.append(task_dict)


@step("construct-catwalk-model", version="002")
def construct_catwalk_model(model: str, model_path: Optional[str] = None, model_class: Optional[str] = None) -> Model:
    # TODO: ugly. clean up later.
    if model not in MODELS:
        prefix_split = model.split("::", 1)
        model_name = prefix_split[-1]
        # prefix = "" if len(prefix_split) == 1 else prefix_split[0]+"::"
        model_args = simple_parse_args_string(model_name)
        if 'pretrained' not in model_args:
            raise ValueError(f"Unknown model {model}")
        hf_name = model_args['pretrained']
        del model_args['pretrained']
        if model_path:
            hf_name = model_path
        if model_class:
            model_args['model_class'] = locate(model_class)

            # TODO: why do we do this?
            # # Assuming tokenizer will be loaded with model, so fail if trying to load it otherwise
            # model_args['pretrained_tokenizer_name_or_path'] = 'UnknownTokenizer'
            model_args['pretrained_tokenizer_name_or_path'] = model_path

        add_decoder_only_model(model_name, hf_name, **model_args)
    return MODELS[model]
