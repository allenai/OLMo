from abc import ABC
from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import efficiency_benchmark.metrics
import torchmetrics
from efficiency_benchmark.metrics.bleu import BLEUMetric
from efficiency_benchmark.metrics.entropy import EntropyMetric
from efficiency_benchmark.metrics.perplexity import PerplexityMetric
from efficiency_benchmark.tango_utils import det_hash
from mypy_extensions import KwArg

PERPLEXITY_METRICS = {
    "word_perplexity": PerplexityMetric,
    "byte_perplexity": PerplexityMetric,
    "bits_per_byte": EntropyMetric,
}

MT_METRICS = {"bleu": BLEUMetric}

QA_METRICS = {
    "squad_metrics": torchmetrics.SQuAD,
}


try:
    from functools import cache as memoize
except ImportError:

    def memoize(user_function, /):  # type: ignore
        import functools

        return functools.lru_cache(maxsize=None)(user_function)


@memoize
def mc_metrics(num_classes: int):
    return {
        "acc": efficiency_benchmark.metrics.AccuracyMetric,
    }


@memoize
def classification_metrics(num_classes: int):
    return {
        "acc": efficiency_benchmark.metrics.AccuracyMetric,
    }


ENTAILMENT_METRICS = classification_metrics(2)

BINARY_CLASSIFICATION_METRICS = classification_metrics(2)


class InstanceFormat(Enum):
    HF_DICT = 1
    HF_MC = 2
    ELEUTHER_DOC = 3
    ELEUTHER_CONTEXT = 4
    ELEUTHER_REQUESTS = 5
    T5_PROMPT = 6
    RANK_CLASSIFICATION = 7
    HF_QA = 8
    PROMPTSOURCE = 9
    HF_CLASSIFICATION = 10
    EFFICIENCY_BENCHMARK = 11


@dataclass
class RankClassificationInstance:
    choices: List[Tuple[str, str]]
    correct_choice: Optional[int]


InstanceConversion = Union[Callable[[Dict[str, Any]], Any], Callable[[Dict[str, Any], KwArg()], Any]]


class Task(ABC):
    """
    Base class for tasks in Catwalk
    """

    def __init__(self, *, version_override: Optional[str] = None):
        if version_override is not None:
            self.VERSION = version_override
        self.metrics: Dict[str, Callable[[], torchmetrics.Metric]] = {}
        self.instance_conversions: Dict[InstanceFormat, InstanceConversion] = {}

    def det_hash_object(self) -> Any:
        # It sucks that we need this. We have to fix this in Tango.
        if hasattr(self, "__getstate__"):
            return self.__getstate__()
        else:
            return self.__dict__

    def has_split(self, split: str) -> bool:
        """Returns ``True`` if a split with the given name exists. ``False`` otherwise."""
        raise NotImplementedError

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        """Returns the split with the given name."""
        raise NotImplementedError

    @property
    def default_split(self) -> str:
        """Returns the name of the default split to run evaluations on."""
        return "test"

    @property
    def fewshot_instances_split(self) -> str:
        """Returns the name of the split to use to find few-shot instances in."""
        for split_name in ["train", "training", "validation"]:
            if self.has_split(split_name):
                return split_name
        raise ValueError("This task has no split to take fewshot instances from.")

    def make_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {name: metric_fn() for name, metric_fn in self.metrics.items()}

    def has_instance_conversion(self, format: InstanceFormat) -> bool:
        return format in self.instance_conversions

    def convert_instance(self, instance: Dict[str, Any], format: InstanceFormat, **kwargs):
        return self.instance_conversions[format](instance, **kwargs)

    def get_fewshot_instances(
        self,
        num_shots: int,
        *,
        exceptions: Union[None, Dict[str, Any], Iterable[Dict[str, Any]]] = None,
        random_seed: int = 18830087,
    ) -> Sequence[Dict[str, Any]]:
        if num_shots <= 0:
            return []

        if exceptions is None:
            exceptions = []
        elif isinstance(exceptions, Dict):
            exceptions = [exceptions]
        exceptions = frozenset(det_hash(e) for e in exceptions)

        r = Random(random_seed)
        instances = self.get_split(self.fewshot_instances_split)
        sampled_instances = [
            instance
            for instance in r.sample(instances, num_shots + len(exceptions))
            if det_hash(instance) not in exceptions
        ]
        return sampled_instances[:num_shots]

    #
    # builder-style methods
    #

    def add_metric(self, name: str, metric: Callable[[], torchmetrics.Metric]):
        if isinstance(metric, torchmetrics.Metric):
            # Users should not do this, but they will, so we try to handle it.
            metric_object = metric.clone()
            metric_object.reset()
            self.metrics[name] = metric_object.clone
        else:
            self.metrics[name] = metric
        return self

    def add_metrics(self, metrics: Mapping[str, Callable[[], torchmetrics.Metric]]):
        for name, metric_fn in metrics.items():
            self.add_metric(name, metric_fn)
        return self

    def add_instance_conversion(self, format: InstanceFormat, conversion: InstanceConversion):
        self.instance_conversions[format] = conversion
        return self


class WithAnswerOptionsMixin:
    def __init__(self, answer_options: Sequence[str]):
        self.answer_options = answer_options
