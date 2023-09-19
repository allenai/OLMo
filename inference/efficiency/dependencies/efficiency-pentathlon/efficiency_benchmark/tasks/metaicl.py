import functools
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import datasets
from efficiency_benchmark.tango_utils import MappedSequence, det_hash
from efficiency_benchmark.task import InstanceFormat, RankClassificationInstance, Task


class MetaICLTask(Task):
    """A task that loads data in the MetaICL fewshot setting. This uses the same set of ICL demonstrations for all test instances."""

    def __init__(self, dataset_name: str, *, version_override: Optional[str] = None):
        super().__init__(version_override=version_override)
        self.dataset_name = dataset_name
        self.add_instance_conversion(InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification)

    def has_split(self, split: str) -> bool:
        return split in ["test"]

    @property
    def fewshot_instances_split(self) -> str:
        """Returns the name of the split to use to find few-shot instances in."""
        raise NotImplementedError(
            "MetaICL uses a fixed set of ICL demonstrations rather than sampling from a split"
        )

    @functools.lru_cache
    def _get_dataset(self, num_shots: int, fewshot_seed: int, split: str):
        data_files = {
            split: f"data/{self.dataset_name}/{self.dataset_name}_{num_shots}_{fewshot_seed}_{split}.jsonl"
        }
        return datasets.load_dataset("allenai/metaicl-data", data_files=data_files, split=split)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        assert self.has_split(split)
        ds = self._get_dataset(num_shots=16, fewshot_seed=100, split=split)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds

    def get_fewshot_instances(
        self,
        num_shots: int,
        *,
        exceptions: Union[None, Dict[str, Any], Iterable[Dict[str, Any]]] = None,
        random_seed: int = 100,
    ) -> Sequence[Dict[str, Any]]:
        if num_shots == 0:
            return []
        assert random_seed in [100, 13, 21, 42, 87] and num_shots <= 16, "Only prebuilt seeds supported for now"

        # For now we only have 16 shot cached so we just subsample it
        subsample_num_shots = num_shots
        num_shots = 16

        if exceptions is None:
            exceptions = []
        elif isinstance(exceptions, Dict):
            exceptions = [exceptions]
        exceptions = frozenset(det_hash(e) for e in exceptions)

        ds = self._get_dataset(num_shots=num_shots, fewshot_seed=random_seed, split="train")
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)

        assert len(ds) == num_shots
        assert not any(
            det_hash(instance) in exceptions for instance in ds
        ), "MetaICL should never have overlap between inference and fewshot splits"

        return ds[:subsample_num_shots]

    def instance_as_rank_classification(
        self,
        instance: Dict[str, Any],
        *,
        fewshot_instances: Optional[Sequence[Dict[str, Any]]] = None,
        continuation_seperator: str = " ",
        example_seperator: str = "\n\n",
        **kwargs,
    ) -> RankClassificationInstance:
        if fewshot_instances is None:
            fewshot_instances = []
        prefix = ""
        for fewshot_instance in fewshot_instances:
            as_rc = self.instance_as_rank_classification(
                fewshot_instance,
                continuation_seperator=continuation_seperator,
                example_seperator=example_seperator,
            )
            if as_rc.correct_choice is None:
                raise ValueError("Could not determine correct choice in ranked classification instance.")
            correct_choice = as_rc.choices[as_rc.correct_choice]
            prefix += correct_choice[0] + correct_choice[1] + example_seperator

        choices = [(prefix + instance["input"], continuation_seperator + option) for option in instance["options"]]

        label = instance["options"].index(instance["output"])
        assert label < len(choices)
        return RankClassificationInstance(choices, label)
