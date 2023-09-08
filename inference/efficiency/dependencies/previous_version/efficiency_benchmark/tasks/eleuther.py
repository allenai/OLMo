import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

from efficiency_benchmark.dependencies.lm_eval.base import Task as EAITask
from efficiency_benchmark.dependencies.lm_eval.tasks import get_task as get_eai_task
from efficiency_benchmark.tango_utils import MappedSequence
from efficiency_benchmark.task import (
    InstanceFormat,
    RankClassificationInstance,
    Task,
    WithAnswerOptionsMixin,
    classification_metrics,
)

T = TypeVar("T")


def _identity(x: T) -> T:
    return x


class EleutherTask(Task):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        version_override: Optional[str] = None,
        ranked_classification: bool = False,
    ):
        Task.__init__(self, version_override=version_override)
        self.eleuther_task: Optional[EAITask] = None
        self.id2label_dict: Dict[int, str] = None
        if isinstance(eleuther_task, str):
            # Eleuther tasks eagerly download their data when they are created. We don't want that, so we have to
            # make this lazy.
            self.eleuther_task_fn = get_eai_task(eleuther_task)
            self.dataset_name = self.eleuther_task_fn.DATASET_NAME
            self.dataset_path = self.eleuther_task_fn.DATASET_PATH
            self.eleuther_task = None
        else:
            self.eleuther_task_fn = eleuther_task
            self.eleuther_task = eleuther_task()
            self.dataset_name = self.eleuther_task.DATASET_NAME
            self.dataset_path = self.eleuther_task.DATASET_PATH
        # Sometimes the "path" is a path to a Python file. We have to fix that.
        self.dataset_path = os.path.splitext(os.path.basename(self.dataset_path))[0]

        self.add_instance_conversion(InstanceFormat.HF_DICT, _identity)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_DOC, self.instance_as_eleuther_doc)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_CONTEXT, self.instance_to_eleuther_context)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_REQUESTS, self.instance_as_eleuther_requests)
        if ranked_classification:
            self.add_instance_conversion(InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification)

    def __getstate__(self):
        result = self.__dict__.copy()
        result["eleuther_task"] = None  # We just cache this, so it doesn't need to be serialized.
        return result

    @property
    def inner_task(self) -> EAITask:
        if self.eleuther_task is None:
            self.eleuther_task = self.eleuther_task_fn()
        return self.eleuther_task

    def id2label(self, id):
        if self.id2label_dict is None:
            split = list(self.inner_task.dataset.keys())[0]
            int2str = self.inner_task.dataset[split].features["label"]._int2str
            self.id2label_dict = {i: int2str[i] for i in range(len(int2str))}
        return self.id2label_dict[id]

    def has_split(self, split: str) -> bool:
        return split in self.inner_task.dataset

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.inner_task.dataset[split]
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(_identity, ds)
        return ds

    @property
    def default_split(self) -> str:
        # In EAI, this is different than `has_split`.
        if self.inner_task.has_test_docs():
            return "test"
        elif self.inner_task.has_validation_docs():
            return "validation"
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs.")

    def instance_as_eleuther_doc(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        return self.inner_task._process_doc(instance)

    def instance_to_eleuther_context(self, instance: Dict[str, Any], *, num_fewshot: int = 0) -> str:
        return self.inner_task.fewshot_context(self.instance_as_eleuther_doc(instance), num_fewshot, rnd=random)

    def instance_as_eleuther_requests(self, instance: Dict[str, Any], *, num_fewshot: int = 0):
        context = self.instance_to_eleuther_context(instance, num_fewshot=num_fewshot)
        return self.inner_task.construct_requests(self.instance_as_eleuther_doc(instance), context)

    def _guess_label(self, instance: Dict[str, Any]) -> int:
        doc = self.instance_as_eleuther_doc(instance)
        label = doc.get("label")
        if label is None:
            label = doc.get("gold")
        if label is None:
            label = doc.get("answer")
        if label is None:
            raise ValueError("Could not find label for instance.")

        if isinstance(label, str):
            label = label[0].lower()
            try:
                label = int(label) - 1
            except ValueError:
                label = ord(label) - ord("a")

        if not isinstance(label, int):
            raise ValueError("Could not find label for instance.")

        return label

    def instance_as_rank_classification(
        self, instance: Dict[str, Any], *, fewshot_instances: Optional[List[Dict[str, Any]]] = None, **kwargs
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

        requests = self.instance_as_eleuther_requests(instance, **kwargs)
        choices = [(prefix + r.args[0], r.args[1]) for r in requests]

        label = self._guess_label(instance)
        assert label < len(choices)
        return RankClassificationInstance(choices, label)


class EleutherClassificationTask(EleutherTask, WithAnswerOptionsMixin):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        answer_options: Sequence[str],
        version_override: Optional[str] = None,
    ):
        EleutherTask.__init__(self, eleuther_task, version_override=version_override, ranked_classification=True)
        WithAnswerOptionsMixin.__init__(self, answer_options)
        self.add_instance_conversion(InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification)
        self.add_metrics(classification_metrics(len(answer_options)))

    def instance_as_rank_classification(
        self, instance: Dict[str, Any], *, fewshot_instances: Optional[List[Dict[str, Any]]] = None, **kwargs
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

        requests = self.instance_as_eleuther_requests(instance, **kwargs)
        choices = [(prefix + r.args[0], r.args[1]) for r in requests]
        assert len(choices) == len(self.answer_options)

        # Reorder the choices so they correspond to self.answer_options.
        # This is important because otherwise doc.label does not match.
        normalized_answer_to_choice = {
            continuation.strip().lower(): (context, continuation) for context, continuation in choices
        }
        choices = [
            normalized_answer_to_choice[answer_option.strip().lower()] for answer_option in self.answer_options
        ]

        label = self._guess_label(instance)
        assert label < len(choices)
        return RankClassificationInstance(choices, label)


class RaceEleutherTask(EleutherTask):
    """The EAI Race task is different because there is no 1:1 correspondence between HF instances and EAI
    instances. EAI chose to follow the GPT3 evaluation approach, which combines multiple questions into one."""

    def __init__(self, *, version_override: Optional[str] = None):
        super().__init__("race", version_override=version_override)
        del self.instance_conversions[InstanceFormat.HF_DICT]
        self.add_instance_conversion(InstanceFormat.ELEUTHER_DOC, lambda x: x)

    def has_split(self, split: str) -> bool:
        if split == "train":
            return self.inner_task.has_training_docs()
        if split == "test":
            return self.inner_task.has_test_docs()
        if split == "validation":
            return self.inner_task.has_validation_docs()
        return False

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        if split == "train":
            return self.inner_task.training_docs()
        if split == "test":
            return self.inner_task.test_docs()
        if split == "validation":
            return self.inner_task.validation_docs()
        raise KeyError(split)


class EleutherTaskWithRenamedSplits(EleutherTask):
    """This task is different because EAI relabels the datasets."""

    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        version_override: Optional[str] = None,
        ranked_classification: bool = False,
    ):
        super().__init__(
            eleuther_task, version_override=version_override, ranked_classification=ranked_classification
        )

    def has_split(self, split: str) -> bool:
        if split == "train":
            return self.inner_task.has_training_docs()
        if split == "test":
            return self.inner_task.has_test_docs()
        if split == "validation":
            return self.inner_task.has_validation_docs()
        return False

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        if split == "train":
            result = self.inner_task.training_docs()
        elif split == "test":
            result = self.inner_task.test_docs()
        elif split == "validation":
            result = self.inner_task.validation_docs()
        else:
            raise KeyError(split)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        return MappedSequence(lambda x: x, result)


class EleutherClassificationTaskWithRenamedSplits(EleutherTaskWithRenamedSplits, WithAnswerOptionsMixin):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        answer_options: Sequence[str],
        version_override: Optional[str] = None,
    ):
        EleutherTaskWithRenamedSplits.__init__(
            self, eleuther_task, version_override=version_override, ranked_classification=True
        )
        WithAnswerOptionsMixin.__init__(self, answer_options)
        self.add_instance_conversion(InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification)
        self.add_metrics(classification_metrics(len(answer_options)))

    instance_as_rank_classification = EleutherClassificationTask.instance_as_rank_classification
