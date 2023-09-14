import functools
import os
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Union

import datasets
import numpy as np
from datasets import Dataset
from efficiency_benchmark.tango_utils import MappedSequence
from efficiency_benchmark.task import InstanceConversion, Task
from efficiency_benchmark.tasks import InstanceFormat
from efficiency_benchmark.tasks.huggingface import get_from_dict

NUM_SINGLE_STREAM_INSTANCES = 1000
NUM_RANDOM_BATCH_INSTANCES = 4000
NUM_OFFLINE_INSTANCES = 8000


@dataclass
class EfficiencyBenchmarkInstance:
    input: Union[str, Dict[str, Any]]
    target: Optional[str]
    id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        d = {"input": self.input}
        if self.target is not None:
            d["target"] = self.target
        if self.id is not None:
            d["id"] = self.id
        return d


class EfficiencyBenchmarkTask(Task):
    def __init__(
        self, dataset_path: str, dataset_name: Optional[str] = None, *, version_override: Optional[str] = None
    ):
        Task.__init__(self, version_override=version_override)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.online_instances: List[EfficiencyBenchmarkInstance] = None

    def has_split(self, split: str) -> bool:
        return split in datasets.get_dataset_split_names(self.dataset_path, self.dataset_name)

    def base_dir(self, base_dir: str) -> str:
        return os.path.join(base_dir, self.dataset_path, self.dataset_name)

    def offline_data_path(self, base_dir: str) -> str:
        return os.path.join(self.base_dir(base_dir), "offline", "data.json")

    def offline_output_path(self, base_dir: str) -> str:
        return os.path.join(self.base_dir(base_dir), "offline", "outputs.json")

    def _convert_instances(self, instances: Sequence[Dict[str, Any]], instance_format) -> MappedSequence:
        return MappedSequence(self.instance_conversions[instance_format], instances)

    def load_instances_from_json(self, path: str) -> List[EfficiencyBenchmarkInstance]:
        return Dataset.from_json(path).to_list()

    def save_instances_to_json(self, instances: List[EfficiencyBenchmarkInstance], path: str):
        instances = [i.to_dict() for i in instances]
        Dataset.from_list(instances).to_json(path)
        return

    def get_instances(self, split: str, num_instances: Optional[int] = None) -> List[EfficiencyBenchmarkInstance]:
        instances: List[EfficiencyBenchmarkInstance] = None
        if self.online_instances is not None:
            instances = self.online_instances
        else:
            instances = self.get_split(split=split)
            instances = list(self._convert_instances(instances, InstanceFormat.EFFICIENCY_BENCHMARK))
            self.online_instances = instances

        def _maybe_extend_and_shuffle(_instances) -> List[EfficiencyBenchmarkInstance]:
            if num_instances is not None:
                while len(_instances) < num_instances:
                    _instances.extend(self.online_instances)
                if len(_instances) > num_instances:
                    _instances = Random(0).sample(_instances, k=num_instances)
            Random(42).shuffle(_instances)
            return _instances

        return _maybe_extend_and_shuffle(instances)

    def get_single_stream_instances(self, split: str) -> List[EfficiencyBenchmarkInstance]:
        return self.get_instances(split=split, num_instances=NUM_SINGLE_STREAM_INSTANCES)

    def get_random_batch_instances(self, split: str) -> List[EfficiencyBenchmarkInstance]:
        return self.get_instances(split=split, num_instances=NUM_RANDOM_BATCH_INSTANCES)

    def prepare_offline_instances(self, base_dir: str, split: str, override: bool = True) -> None:
        path: str = self.offline_data_path(base_dir)
        if os.path.exists(path) and not override:
            print(f"Offline instances already exist: {path}. Skipping...")
            return
        instances = self.get_instances(split=split, num_instances=NUM_OFFLINE_INSTANCES)
        try:
            # Try to cache preprocessed instances to a file
            self.save_instances_to_json(instances, path)
            print(f"Saved offline instances to {path}.")
        except:
            print(f"Failed to save offline instances to file: {path}")

    def get_scenario_instances(self, scenario: str, split: str) -> List[EfficiencyBenchmarkInstance]:
        funcs = {
            "single_stream": self.get_single_stream_instances,
            "random_batch": self.get_random_batch_instances,
            "accuracy": self.get_instances,
        }
        return funcs[scenario](split=split)

    def dataset(self, split: str):
        return datasets.load_dataset(self.dataset_path, self.dataset_name, split=split)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.dataset(split=split)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds


# class EfficiencyBenchmarkTranslationTask(EfficiencyBenchmarkTask):
#     def __init__(
#         self,
#         dataset_path: str,
#         dataset_name: Optional[str] = None,
#         *,
#         version_override: Optional[str] = None
#     ):
#         EfficiencyBenchmarkTask.__init__(self, dataset_path, dataset_name, version_override=version_override)


# class EfficiencyBenchmarkClassificationTask(EfficiencyBenchmarkTask):
#     def __init__(
#         self,
#         dataset_path: str,
#         dataset_name: Optional[str] = None,
#         *,
#         version_override: Optional[str] = None
#     ):
#         EfficiencyBenchmarkTask.__init__(self, dataset_path, dataset_name, version_override=version_override)


class EfficiencyBenchmarkPromptTask(EfficiencyBenchmarkTask):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None,
    ):
        EfficiencyBenchmarkTask.__init__(self, dataset_path, dataset_name)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.dataset(split=split)
        cleaned_data = []
        for instance in ds:
            if len(instance["text"]) < 2:
                continue
            cleaned_data.append(instance)
        ds = Dataset.from_list(cleaned_data)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds


class EfficiencyBenchmarkRaftTask(EfficiencyBenchmarkTask):
    def __init__(self, subset: str):
        EfficiencyBenchmarkTask.__init__(self, "ought/raft", subset)


def efficiency_benchmark_mt_conversion(**kwargs) -> InstanceConversion:
    def convert(
        instance: Dict[str, Any], *, input_field: str, target_field: str, id_field: Optional[str] = None
    ) -> EfficiencyBenchmarkInstance:
        instance = instance["translation"]
        input = get_from_dict(instance, input_field)
        target = get_from_dict(instance, target_field)
        return EfficiencyBenchmarkInstance(
            id=str(get_from_dict(instance, id_field)) if id_field else None, input=input, target=target
        )

    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(convert, **kwargs)


def efficiency_benchmark_classification_conversion(
    **kwargs,
) -> InstanceConversion:
    def convert(
        instance: Dict[str, Any],
        *,
        label_map: Dict[int, str],
        premise_field: str = "premise",
        hypothesis_field: Optional[str] = "hypothesis",
        label_field: str = "label",
        id_field: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> EfficiencyBenchmarkInstance:
        input = {premise_field: get_from_dict(instance, premise_field)}
        if hypothesis_field is not None:
            input[hypothesis_field] = get_from_dict(instance, hypothesis_field)
        if task_name:
            input["task_name"] = task_name

        label_id = int(get_from_dict(instance, label_field))
        target = label_map[label_id]
        return EfficiencyBenchmarkInstance(
            input=input, target=target, id=str(get_from_dict(instance, id_field)) if id_field else None
        )

    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(convert, **kwargs)


def efficiency_benchmark_raft_conversion(
    **kwargs,
) -> InstanceConversion:
    def convert(
        instance: Dict[str, Any],
        *,
        label_field: str = "Label",
        id_field: Optional[str] = "ID",
        task_name: Optional[str] = None,
    ) -> EfficiencyBenchmarkInstance:
        input = instance
        if task_name:
            input["task_name"] = task_name
        if label_field in input:
            input.pop(label_field)
        return EfficiencyBenchmarkInstance(
            input=input, target=None, id=str(get_from_dict(instance, id_field)) if id_field else None
        )

    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(convert, **kwargs)


def efficiency_benchmark_prompt_conversion(
    **kwargs,
) -> InstanceConversion:
    def convert(instance: Dict[str, Any], max_length: int = 512) -> EfficiencyBenchmarkInstance:
        text = instance["text"]
        random_length = np.random.randint(max_length)
        text = " ".join(text.split()[:random_length])
        return EfficiencyBenchmarkInstance(input=text, target=None, id=None)

    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(convert, **kwargs)
