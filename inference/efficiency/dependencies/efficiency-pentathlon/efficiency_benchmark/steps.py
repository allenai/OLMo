import itertools
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import more_itertools
import numpy as np
import torch
from datasets import Dataset
from efficiency_benchmark.efficiency.profiler import Profiler
from efficiency_benchmark.stdio_wrapper import StdioWrapper
from efficiency_benchmark.task import Task
from efficiency_benchmark.tasks import TASKS, EfficiencyBenchmarkTask
from efficiency_benchmark.tasks.efficiency_benchmark import EfficiencyBenchmarkInstance

from hf_olmo import *

olmo_checkpoint = "/net/nfs.cirrascale/allennlp/akshitab/olmo-models/olmo-1b/"
tokenizer = OLMoTokenizerFast.from_pretrained(olmo_checkpoint)
EXPECTED_BATCH_SIZE = 128
NUM_BATCHES = 1000


class PredictStep:
    def __init__(
        self,
        *,
        cmd: List[str],
        task: str,
        scenario: str,
        max_batch_size: int,
        offline_dir: str,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ):
        np.random.seed(42)
        self.task: EfficiencyBenchmarkTask = TASKS[task]
        if scenario == "offline" and "raft" not in task:
            # We prefer training split for the offline scenario, which usually has more data
            # RAFT's training data is tiny, thus we use the test split for the offline scenario
            self.split = "train"
        else:
            self.split = split if split is not None else self.task.default_split

        self.scenario = scenario
        self.max_batch_size = max_batch_size
        self.offline_dir = offline_dir
        self.limit = limit
        self.cmd = cmd
        self.predictor = StdioWrapper(cmd=cmd)
        self.profiler = Profiler(interval=0.1)
        self.targets = None

        self._prepare_data()

    def _prepare_data(self):
        if self.scenario == "offline":
            self.task.prepare_offline_instances(base_dir=self.offline_dir, split=self.split)
            self.offline_data_path = self.task.offline_data_path(base_dir=self.offline_dir)
            self.offline_output_path = self.task.offline_output_path(base_dir=self.offline_dir)
            return

        instances: List[EfficiencyBenchmarkInstance] = self.task.get_scenario_instances(
            scenario=self.scenario, split=self.split
        )
        if self.limit is not None and self.limit > 0 and len(instances) > self.limit:
            indices = np.random.choice(list(range(len(instances))), size=self.limit, replace=False)
            instances = [instances[i] for i in indices]
        self.num_instances = len(instances)

        if self.scenario == "accuracy":
            batches = list(more_itertools.chunked(instances, self.max_batch_size))
        elif self.scenario == "single_stream":
            batches = list(more_itertools.chunked(instances, 1))
        elif self.scenario == "random_batch":
            num_instances_per_batch = np.random.poisson(lam=EXPECTED_BATCH_SIZE, size=NUM_BATCHES)
            batches = []
            idx = 0
            for n in num_instances_per_batch:
                if n == 0:
                    continue
                batch = instances[idx : idx + n]
                idx += n
                batches.append(batch)
                if idx >= len(instances):
                    break
        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}. Choose from 'single_stream', 'random_batch', 'offline'"
            )

        self.num_batches = len(batches)
        assert self.num_instances == sum(len(batch) for batch in batches)

        self.input_batches = [[instance.input for instance in batch] for batch in batches]
        if self.scenario == "accuracy":
            target_batches = [[instance.target for instance in batch] for batch in batches]
            self.targets = list(itertools.chain(*target_batches))
            assert len(self.targets) == self.num_instances

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        if kwargs["split"] is None:
            kwargs["split"] = kwargs["task"].default_split
        return kwargs

    def tabulate_efficiency_metrics(
        self,
        metrics: Dict[str, Any],
        output_file: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        print(f"Time Elapsed: {metrics['time']:.2f} s")
        # print(f"Max DRAM Memory Usage: {max_mem_util * total_memory: .2f} GiB")
        # print(f"Number of Parameters: {efficiency_metrics['num_params'] / 1e6: .2f} M")
        print(f"Max GPU memory usage: {metrics['max_gpu_mem']: .2f} GiB.")
        # print(f"GPU Energy: {metrics['gpu_energy']:.2e} Wh")
        # print(f"CPU Energy: {metrics['cpu_energy']: .2e} Wh")
        # print(f"Memory Energy: {metrics['dram_energy']: .2e} Wh")
        print(f"Average GPU power: {metrics['avg_gpu_power']: .2e} W.")
        print(f"Average power: {metrics['avg_power']: .2e} W.")
        print(f"Total energy: {metrics['total_energy']: .2e} kWh.")
        print(f"CO2 emission: {metrics['carbon']: .2e} kg.")
        print(f"Throughput: {metrics['throughput']: .2f} instances / s.")
        print(f"Throughput (words): {metrics['throughput_words']: .2f} words / s.")
        print(f"Throughput (tokens): {metrics['throughput_tokens']: .2f} tokens / s.")
        if self.scenario != "offline":
            metrics["latency"] = metrics["time"] / self.num_batches
            print(f"Latency: {metrics['latency'] * 1000: .2f} ms / batch.")

        if output_file is not None:
            try:
                with open(output_file, "w") as fout:
                    json.dump(metrics, fout)
            except:
                print(f"Failed to write efficiency metrics to {output_file}.")

    def run(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        self.predictor.start()
        if self.scenario == "offline":
            return self.run_offline()
        else:
            return self.run_online()

    def run_online(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        output_batches = []
        _ = self.predictor.dummy_predict(dummy_inputs=self.input_batches[-1], max_batch_size=self.max_batch_size)

        self.profiler.start()
        for output_batch in self.predictor.predict(
            input_batches=self.input_batches, max_batch_size=self.max_batch_size
        ):
            output_batches.append(output_batch)
        efficiency_metrics = self.profiler.stop()
        results, num_output_words, num_output_tokens = self.process(output_batches)

        efficiency_metrics["throughput"] = self.num_instances / efficiency_metrics["time"]
        efficiency_metrics["throughput_words"] = num_output_words / efficiency_metrics["time"]
        efficiency_metrics["throughput_tokens"] = num_output_tokens / efficiency_metrics["time"]
        return results, efficiency_metrics

    def run_offline(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        self.predictor.provide_offline_configs(
            offline_data_path=self.offline_data_path,
            offline_output_file=self.offline_output_path,
            limit=self.limit,
        )
        self.profiler.start()
        self.predictor.block_for_prediction()

        efficiency_metrics = self.profiler.stop()
        self.predictor.block_for_outputs()
        self.predictor.stop()
        results = Dataset.from_json(self.offline_output_path).to_list()
        print(f"Loading outputs from {self.offline_output_path}")
        self.num_instances = len(results)
        efficiency_metrics["throughput"] = self.num_instances / efficiency_metrics["time"]
        num_output_words = sum([len(result["output"].split()) for result in results])
        efficiency_metrics["throughput_words"] = num_output_words / efficiency_metrics["time"]
        return results, efficiency_metrics

    def process(self, output_batches: Iterable[str]) -> Tuple[Sequence[Dict[str, Any]], int]:
        yielded_label_index = -1
        results = []
        num_output_words = 0
        num_output_tokens = 0
        for output in output_batches:
            yielded_label_index += 1
            output = output.rstrip()
            target = self.targets[yielded_label_index] if self.targets is not None else None
            result = {
                "target": target,
                "output": output,
            }
            if self.scenario == "accuracy":
                result.update({metric_name: (output, target) for metric_name in self.task.metrics.keys()})
            num_output_words += len(output.split())
            num_output_tokens += len(tokenizer.tokenize(output))
            results.append(result)
        return results, num_output_words, num_output_tokens


class CalculateMetricsStep:
    _TorchmetricsResult = Union[torch.Tensor, Dict[str, "_TorchmetricsResult"]]
    _CatwalkResult = Union[float, Dict[str, "_CatwalkResult"]]

    def __init__(self, task: Union[str, Task]):
        self.task = TASKS[task] if isinstance(task, str) else task

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        return kwargs

    def _tensor_args(self, args: Tuple[Any]) -> Tuple[Any, ...]:
        """
        Annoyingly, torchmetrics only supports tensors as input, not raw values. So we have to convert raw values
        into tensors.

        From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        fixed_args: List[Any] = []
        for arg in args:
            if isinstance(arg, (float, int)):
                fixed_args.append(torch.tensor(arg))
            else:
                fixed_args.append(arg)
        return tuple(fixed_args)

    def _unsqueeze_args(self, args: Tuple[Any]) -> Tuple[Any, ...]:
        """
        Further, torchmetrics can't handle single-instance calls when given tensors. It always needs the first
        dimension of the tensors to be the instance dimension. So we add one.

        From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        fixed_args: List[Any] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                fixed_args.append(arg.unsqueeze(0))
            else:
                fixed_args.append(arg)
        return tuple(fixed_args)

    def _recursive_tolist(self, args: _TorchmetricsResult) -> _CatwalkResult:
        """From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        if isinstance(args, dict):
            return {key: self._recursive_tolist(value) for key, value in args.items()}
        else:
            return args.tolist()

    def calculate_metrics(self, predictions: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """From catwalk.
        https://github.com/allenai/catwalk/blob/main/catwalk/model.py
        """
        metrics = self.task.make_metrics()
        for prediction in predictions:
            for metric_name, metric_args in prediction.items():
                try:
                    metric = metrics[metric_name]
                except KeyError:
                    continue
                metric_args = self._tensor_args(metric_args)
                metric_args = self._unsqueeze_args(metric_args)
                metric.update(*metric_args)
        return {metric_name: self._recursive_tolist(metric.compute()) for metric_name, metric in metrics.items()}


class TabulateMetricsStep:
    def __init__(self):
        pass

    def run(self, metrics: Dict[str, Dict[str, float]], format: str = "text") -> Iterable[str]:
        flattend_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        for task_name, task_metrics in metrics.items():
            for metric_name, metric_value in task_metrics.items():
                # if metric_value is a dict, then it's a nested metric
                if isinstance(metric_value, dict):
                    for nested_metric_name, nested_metric_value in metric_value.items():
                        flattend_metrics[task_name][f"{metric_name}.{nested_metric_name}"] = (
                            nested_metric_value.item()
                            if isinstance(nested_metric_value, torch.Tensor)
                            else nested_metric_value
                        )
                else:
                    flattend_metrics[task_name][metric_name] = metric_value

        if format == "text":
            for task_name, task_metrics in flattend_metrics.items():
                for metric_name, metric_value in task_metrics.items():
                    yield f"{task_name}\t{metric_name}\t{metric_value}"
        elif format == "latex":
            raise NotImplementedError()
        else:
            raise AttributeError("At the moment, only the 'text' format is supported.")


class LogOutputStep:
    def __init__(self, task: Union[str, Task], output_file: Optional[str] = None):
        self.task = TASKS[task] if isinstance(task, str) else task
        self.output_file: str = output_file

    def run(self, predictions: Sequence[Dict[str, Any]]) -> None:
        predictions = self.remove_metrics(predictions, additional_fields=["target"])

        def _log_to_stdout():
            for prediction in predictions:
                print(prediction)

        if self.output_file is not None:
            try:
                outputs = Dataset.from_list(predictions)
                outputs.to_json(self.output_file)
            except:
                print("Failed to write output to file. Logging to stdout instead.")
                _log_to_stdout()
        else:
            _log_to_stdout()

    def remove_metrics(
        self, predictions: Sequence[Dict[str, Any]], additional_fields: Optional[Sequence[str]] = None
    ) -> Sequence[Dict[str, Any]]:
        # Remove metrics from the output.
        for prediction in predictions:
            for metric_name in self.task.metrics.keys():
                prediction.pop(metric_name)
        if additional_fields is not None:
            for prediction in predictions:
                for field in additional_fields:
                    prediction.pop(field)
        return predictions
