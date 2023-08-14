from typing import Any, Dict, List, Optional

from efficiency_benchmark.task import InstanceFormat, RankClassificationInstance
from efficiency_benchmark.tasks import HFDatasetsTask


class P3Task(HFDatasetsTask):
    def __init__(
        self,
        dataset_name: str,
        *,
        version_override: Optional[str] = None,
    ):
        super().__init__("bigscience/P3", dataset_name=dataset_name, version_override=version_override)
        self.add_instance_conversion(InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification)

    def instance_as_rank_classification(
        self,
        instance: Dict[str, Any],
        *,
        fewshot_instances: Optional[List[Dict[str, Any]]] = None,
    ) -> RankClassificationInstance:
        if fewshot_instances is None:
            fewshot_instances = []
        prefix = ""
        for fewshot_instance in fewshot_instances:
            as_rc = self.instance_as_rank_classification(fewshot_instance)
            if as_rc.correct_choice is None:
                raise ValueError("Could not determine correct choice in ranked classification instance.")
            correct_choice = as_rc.choices[as_rc.correct_choice]
            prefix += f"{correct_choice[0].strip()} {correct_choice[1].strip()}\n\n"

        prefix += f" {instance['inputs_pretokenized'].strip()}"
        correct_choice = instance["targets_pretokenized"].strip()
        try:
            choices = [choice.strip() for choice in instance["answer_choices"]]
        except KeyError:
            raise ValueError("This instance cannot be converted to rank classification format.")
        return RankClassificationInstance([(prefix, choice) for choice in choices], choices.index(correct_choice))
