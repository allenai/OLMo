import functools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import datasets
from efficiency_benchmark.tango_utils import MappedSequence
from efficiency_benchmark.task import InstanceConversion, InstanceFormat, Task


def get_from_dict(d: Union[Mapping[str, Any], Sequence[Any]], field: str, missing_ok: bool = False) -> Any:
    components = field.split(".", 1)
    if len(components) == 0:
        raise ValueError("get_from_dict() called with empty string.")
    elif isinstance(d, Mapping) and len(components) == 1:
        try:
            return d[components[0]]
        except KeyError:
            if missing_ok:
                return None
            else:
                raise
    elif isinstance(d, Sequence) and len(components) == 1:
        try:
            return d[int(components[0])]
        except IndexError:
            if missing_ok:
                return None
            else:
                raise
    elif isinstance(d, Mapping):
        first, rest = components
        try:
            d2 = d[first]
        except KeyError:
            if missing_ok:
                return None
            else:
                raise
        return get_from_dict(d2, rest, missing_ok)
    elif isinstance(d, Sequence):
        first, rest = components
        try:
            d2 = d[int(first)]
        except IndexError:
            if missing_ok:
                return None
            else:
                raise
        return get_from_dict(d2, rest, missing_ok)
    else:
        raise ValueError()


class HFDatasetsTask(Task):
    def __init__(
        self, dataset_path: str, dataset_name: Optional[str] = None, *, version_override: Optional[str] = None
    ):
        Task.__init__(self, version_override=version_override)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.add_instance_conversion(InstanceFormat.HF_DICT, lambda x: x)

    @functools.lru_cache
    def has_split(self, split: str) -> bool:
        return split in datasets.get_dataset_split_names(self.dataset_path, self.dataset_name)

    @functools.lru_cache
    def dataset(self, split: str):
        return datasets.load_dataset(self.dataset_path, self.dataset_name, split=split)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.dataset(split=split)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds


@dataclass
class HFQAInstance:
    id: str
    question: str
    context: str
    answers: List[str]


def hfqa_conversion(
    *,
    context_field: str = "context",
    question_field: str = "question",
    answers_field: str = "answers",
    id_field: str = "id",
) -> InstanceConversion:
    def convert(instance: Dict[str, Any]) -> HFQAInstance:
        return HFQAInstance(
            id=get_from_dict(instance, id_field),
            context=get_from_dict(instance, context_field),
            question=get_from_dict(instance, question_field).strip(),
            answers=get_from_dict(instance, answers_field),
        )

    return convert


@dataclass
class HFMCInstance:
    id: Optional[str]
    question: str
    answer_choices: List[str]
    correct_answer_index: Optional[int]


def normalize_answers(answer: Any, answer_mappings: Optional[Dict[str, int]] = None) -> int:
    if answer_mappings is None:
        if isinstance(answer, int):
            return answer
        if isinstance(answer, str):
            if len(answer) == 1:
                answer = answer.lower()
                answer_index = ord(answer[0])
                if ord("a") <= answer_index <= ord("z"):
                    return answer_index - ord("a")
                # We don't automatically convert str to int because sometimes they are 1-based and sometimes
                # they are 0-based.
            raise ValueError(f"Don't know how to make an index from answer '{answer}'.")
        raise ValueError(f"Don't know how to make an index from answer of type {answer.__class__}.")
    else:
        return answer_mappings[answer]


def hfmc_convert(
    instance: Dict[str, Any],
    *,
    context_field: Optional[str] = None,
    question_field: str,
    answer_choices_fields: Union[str, List[str]],
    correct_answer_index_field: Optional[str] = None,
    correct_answer_field: Optional[str] = None,
    id_field: Optional[str] = None,
    answer_mappings: Optional[Dict[str, int]] = None,
) -> HFMCInstance:
    if isinstance(answer_choices_fields, str):
        answer_choices = get_from_dict(instance, answer_choices_fields)
    else:
        answer_choices = [get_from_dict(instance, field, missing_ok=True) for field in answer_choices_fields]
        answer_choices = [a for a in answer_choices if a is not None]
        assert len(answer_choices) > 0
    answer_choices = [a.strip() for a in answer_choices]

    question = get_from_dict(instance, question_field).strip()
    if context_field is not None:
        question = get_from_dict(instance, context_field).strip() + " " + question

    correct_answer_index: Optional[int]
    if correct_answer_index_field is not None:
        correct_answer = get_from_dict(instance, correct_answer_index_field)
        if correct_answer != "":
            correct_answer_index = normalize_answers(correct_answer, answer_mappings)
        else:
            correct_answer_index = None
    elif correct_answer_field is not None:
        correct_answer_index = answer_choices_fields.index(correct_answer_field)
        # When the correct answer is always given in a field, we have to shuffle the answer options. Otherwise the
        # answer is always the same.
        rng = random.Random(sum(ord(c) for c in question))  # same question always gets the same order
        order = list(range(len(answer_choices)))
        rng.shuffle(order)
        answer_choices = [answer_choices[i] for i in order]
        correct_answer_index = order.index(correct_answer_index)
    else:
        raise RuntimeError(
            "When constructing an hfmc conversion, you have to specify either correct_answer_index_field or correct_answer_field."
        )
    if correct_answer_index == -1:
        correct_answer_index = None

    return HFMCInstance(
        id=str(get_from_dict(instance, id_field)) if id_field else None,
        question=question,
        answer_choices=answer_choices,
        correct_answer_index=correct_answer_index,
    )


def hfmc_conversion(
    **kwargs,
) -> InstanceConversion:
    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(hfmc_convert, **kwargs)


@dataclass
class HFClassificationInstance:
    task_name: str
    id: Optional[str]
    text: Union[str, Dict[str, str]]
    label: str


def hfclassification_convert(
    instance: Dict[str, Any],
    *,
    task_name: str,
    label_map: Dict[int, str],
    premise_field: str = "premise",
    hypothesis_field: Optional[str] = "hypothesis",
    label_field: str = "label",
    id_field: Optional[str] = None,
) -> HFClassificationInstance:
    text = {premise_field: get_from_dict(instance, premise_field)}
    if hypothesis_field is not None:
        text[hypothesis_field] = get_from_dict(instance, hypothesis_field)
    label_id = int(get_from_dict(instance, label_field))
    label = label_map[label_id]
    return HFClassificationInstance(
        task_name=task_name,
        id=str(get_from_dict(instance, id_field)) if id_field else None,
        text=text,
        label=label,
    )


def hfclassification_conversion(
    **kwargs,
) -> InstanceConversion:
    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(hfclassification_convert, **kwargs)
