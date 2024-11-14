import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class ExtrapolateNConfig:
    path: str
    """
    Path containing the W&B downloaded data and metadata.
    """

    mode: str
    """
    Whether this model is used for fitting the curve ('train') or evaluating the fit ('eval').
    """

    n: int
    """
    The model size (non-embedding parameter count).
    """

    label: str
    """
    A short label for this curve.
    """

    color: str
    """
    The color for this curve.
    """


@dataclass
class FinalConfig:
    paths: List[str]
    """
    Path containing the W&B downloaded data and metadata.
    """

    mode: str
    """
    Whether this model is used for fitting the curve ('train') or evaluating the fit ('eval').
    """

    n: int
    """
    The model size (non-embedding parameter count).
    """

    label: str
    """
    A short label for this curve.
    """

    color: str
    """
    The color for this curve.
    """

    use_last_n_percentage: int = 100
    """
    The percent of data points used. Defaults to 100%.
    """


validation = [
    "c4_en-validation",
    "dolma_books-validation",
    "dolma_common-crawl-validation",
    "dolma_pes2o-validation",
    "dolma_reddit-validation",
    "dolma_stack-validation",
    "dolma_wiki-validation",
    "ice-validation",
    "m2d2_s2orc-validation",
    "pile-validation",
    "wikitext_103-validation",
]

v3_validation = [
    "v3-small-c4_en-validation",
    "v3-small-dolma_books-validation",
    "v3-small-dolma_common-crawl-validation",
    "v3-small-dolma_pes2o-validation",
    "v3-small-dolma_reddit-validation",
    "v3-small-dolma_stack-validation",
    "v3-small-dolma_wiki-validation",
    "v3-small-ice-validation",
    "v3-small-m2d2_s2orc-validation",
    #'v3-small-pile-validation',
    "v3-small-wikitext_103-validation",
]


@dataclass
class DownstreamTaskPrediction:
    task_loss_key: Union[str, List[str]]
    task_accuracy_key: Union[str, List[str]]
    task_minimum: float = 0.25
    task_maximum: float = 1.0

    def get_loss_keys(self):
        return self.task_loss_key if isinstance(self.task_loss_key, list) else [self.task_loss_key]

    def get_accuracy_keys(self):
        return self.task_accuracy_key if isinstance(self.task_accuracy_key, list) else [self.task_accuracy_key]


minimums_rc: Dict[str, float] = {
    "piqa": 0.5,
    "socialiqa": 1 / 3,
    "csqa": 0.2,
}

maximums_rc: Dict[str, float] = {}  # {"mmlu_stem": 0.9, "arc_easy": 0.85}


core_names = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "csqa",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
]
core_small_names = ["hellaswag", "arc_challenge", "piqa", "csqa", "socialiqa"]
mmlu_names = ["mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]

core_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    f"{key}_rc_5shot": DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_rc_5shot_bpb_bpb",
        task_accuracy_key=f"eval/downstream/{key}_rc_5shot_len_norm"
        if key not in ["arc_easy", "winogrande", "boolq"]
        else f"eval/downstream/{key}_rc_5shot_acc",
        task_minimum=minimums_rc.get(key, 0.25),
        task_maximum=maximums_rc.get(key, 1.0),
    )
    for key in core_names
}

core_small_5shot_tasks: Dict[str, DownstreamTaskPrediction] = {
    f"{key}_rc_5shot": DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_rc_5shot_bpb_bpb",
        task_accuracy_key=f"eval/downstream/{key}_rc_5shot_len_norm"
        if key not in ["arc_easy", "winogrande"]
        else f"eval/downstream/{key}_rc_5shot_acc",
        task_minimum=minimums_rc.get(key, 0.25),
        task_maximum=maximums_rc.get(key, 1.0),
    )
    for key in core_small_names
}

mmlu_var_tasks: Dict[str, DownstreamTaskPrediction] = {
    "mmlu_avg_var": DownstreamTaskPrediction(
        task_loss_key=[f"eval/downstream_bpb/{key}_var_bpb_bpb" for key in mmlu_names],
        task_accuracy_key=[f"eval/downstream/{key}_var_len_norm" for key in mmlu_names],
        task_minimum=0.25,
        task_maximum=1.0,  # 0.9,
    )
}

mmlu_subset_var_tasks: Dict[str, DownstreamTaskPrediction] = {
    key: DownstreamTaskPrediction(
        task_loss_key=f"eval/downstream_bpb/{key}_var_bpb_bpb",
        task_accuracy_key=f"eval/downstream/{key}_var_len_norm",
        task_minimum=minimums_rc.get(key, 0.25),
        task_maximum=maximums_rc.get(key, 0.9),
    )
    for key in mmlu_names
}


def get_task_sets(keys):
    if len(keys) == 1:
        if keys[0] == "core":
            keys = core_5shot_tasks.keys()
        elif keys[0] == "mmlu":
            keys = list(mmlu_var_tasks.keys()) + list(mmlu_subset_var_tasks.keys())
        elif keys[0] == "main":
            keys = list(mmlu_var_tasks.keys()) + list(core_small_5shot_tasks.keys())
    return keys


def get_bpb_keys(tasks: Dict[str, DownstreamTaskPrediction]) -> List[str]:
    bpb_keys: List[str] = []
    for _, task in tasks.items():
        if isinstance(task.task_loss_key, list):
            bpb_keys += task.task_loss_key
        else:
            bpb_keys.append(task.task_loss_key)
    return bpb_keys


def get_accuracy_keys(tasks: Dict[str, DownstreamTaskPrediction]) -> List[str]:
    accuracy_keys: List[str] = []
    for _, task in tasks.items():
        if isinstance(task.task_accuracy_key, list):
            accuracy_keys += task.task_accuracy_key
        else:
            accuracy_keys.append(task.task_accuracy_key)
    return accuracy_keys


# Special case for testing with old tokenizer:

downstream_newline = [
    "mmlu_newline_social_sciences_var_len_norm",
    "mmlu_newline_humanities_var_len_norm",
    "mmlu_newline_other_var_len_norm",
    "mmlu_newline_stem_mc_5shot_test_len_norm",
    "mmlu_newline_humanities_mc_5shot_len_norm",
    "mmlu_newline_social_sciences_mc_5shot_len_norm",
    "mmlu_newline_stem_var_len_norm",
    "mmlu_newline_other_mc_5shot_test_len_norm",
    "mmlu_newline_humanities_mc_5shot_test_len_norm",
    "mmlu_newline_stem_mc_5shot_len_norm",
    "mmlu_newline_social_sciences_mc_5shot_test_len_norm",
    "mmlu_newline_other_mc_5shot_len_norm",
    "hellaswag_newline_rc_0shot_len_norm",
    "hellaswag_newline_rc_5shot_len_norm",
    "hellaswag_newline_mc_5shot_acc",
    "winogrande_newline_rc_0shot_acc",
    "winogrande_newline_rc_5shot_acc",
    "winogrande_newline_mc_5shot_acc",
    "piqa_newline_rc_0shot_len_norm",
    "piqa_newline_rc_5shot_len_norm",
    "piqa_newline_mc_5shot_acc",
    "socialiqa_newline_rc_0shot_len_norm",
    "socialiqa_newline_rc_5shot_len_norm",
    "socialiqa_newline_mc_5shot_acc",
    "openbookqa_newline_rc_0shot_len_norm",
    "openbookqa_newline_rc_5shot_len_norm",
    "openbookqa_newline_mc_5shot_acc",
    "csqa_newline_rc_0shot_len_norm",
    "csqa_newline_rc_5shot_len_norm",
    "csqa_newline_mc_5shot_acc",
    "boolq_newline_rc_0shot_acc",
    "boolq_newline_rc_5shot_acc",
    "boolq_newline_mc_5shot_acc",
    "copa_newline_rc_0shot_acc",
    "arc_easy_newline_rc_0shot_acc",
    "arc_easy_newline_rc_5shot_acc",
    "arc_easy_newline_mc_5shot_acc",
    "arc_challenge_newline_rc_0shot_len_norm",
    "arc_challenge_newline_rc_5shot_len_norm",
    "arc_challenge_newline_mc_5shot_acc",
    "sciq_newline_rc_0shot_acc",
]
downstream_newline_bpb = [
    "mmlu_newline_stem_var_bpb",
    "mmlu_newline_humanities_var_bpb",
    "mmlu_newline_social_sciences_var_bpb",
    "mmlu_newline_other_var_bpb",
    "mmlu_newline_stem_bpb",
    "mmlu_newline_humanities_bpb",
    "mmlu_newline_social_sciences_bpb",
    "mmlu_newline_other_bpb",
    "piqa_newline_rc_0shot_bpb",
    "piqa_newline_rc_5shot_bpb",
    "piqa_newline_mc_5shot_bpb",
    "hellaswag_newline_rc_0shot_bpb",
    "hellaswag_newline_rc_5shot_bpb",
    "hellaswag_newline_mc_5shot_bpb",
    "winogrande_newline_rc_0shot_bpb",
    "winogrande_newline_rc_5shot_bpb",
    "winogrande_newline_mc_5shot_bpb",
    "openbookqa_newline_rc_0shot_bpb",
    "openbookqa_newline_rc_5shot_bpb",
    "openbookqa_newline_mc_5shot_bpb",
    "boolq_newline_rc_0shot_bpb",
    "boolq_newline_rc_5shot_bpb",
    "boolq_newline_mc_5shot_bpb",
    "sciq_newline_rc_0shot_bpb",
    # "sciq_newline_rc_5shot_bpb",
    # "sciq_newline_mc_5shot_bpb",
    "arc_easy_newline_rc_0shot_bpb",
    "arc_easy_newline_rc_5shot_bpb",
    "arc_easy_newline_mc_5shot_bpb",
    "arc_challenge_newline_rc_0shot_bpb",
    "arc_challenge_newline_rc_5shot_bpb",
    "arc_challenge_newline_mc_5shot_bpb",
    "copa_newline_rc_0shot_bpb",
    # "copa_newline_rc_5shot_bpb",
    # "copa_newline_mc_5shot_bpb",
    "csqa_newline_rc_0shot_bpb",
    "csqa_newline_rc_5shot_bpb",
    "csqa_newline_mc_5shot_bpb",
    "socialiqa_newline_rc_0shot_bpb",
    "socialiqa_newline_rc_5shot_bpb",
    "socialiqa_newline_mc_5shot_bpb",
]

tasks = {**core_5shot_tasks, **mmlu_var_tasks, **mmlu_subset_var_tasks}
downstream_bpb = get_bpb_keys(tasks)
downstream = get_accuracy_keys(tasks)

KEYS_BY_KEY = {
    "all-val-lm": [f"eval/{val}/CrossEntropyLoss" for val in validation],
    "all-bpb": downstream_bpb,
    "c4": ["eval/c4_en-validation/CrossEntropyLoss"],
}

WEIGHT_BY_KEY = {
    "eval/downstream_bpb/mmlu_stem_var_bpb_bpb": 0.215,
    "eval/downstream_bpb/mmlu_humanities_var_bpb_bpb": 0.335,
    "eval/downstream_bpb/mmlu_social_sciences_var_bpb_bpb": 0.219,
    "eval/downstream_bpb/mmlu_other_var_bpb_bpb": 0.231,
    "eval/downstream/mmlu_stem_var_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_var_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_var_len_norm": 0.219,
    "eval/downstream/mmlu_other_var_len_norm": 0.231,
    "eval/downstream/mmlu_stem_mc_5shot_len_norm": 0.215,
    "eval/downstream/mmlu_humanities_mc_5shot_len_norm": 0.335,
    "eval/downstream/mmlu_social_sciences_mc_5shot_len_norm": 0.219,
    "eval/downstream/mmlu_other_mc_5shot_len_norm": 0.231,
}


# peteish model flops

MODEL_FLOPS = {
    "190m": 1903391232,
    "370m": 3443922944,
    "600m": 5180751744,
    "760m": 6373843968,
    "1b": 10109071360,
    "3b": 22970355200,
    "7b": 49412071424,
    "13b": 91335915520,
}

for task_name, task in tasks.items():
    KEYS_BY_KEY[task_name] = task.task_loss_key if isinstance(task.task_loss_key, list) else [task.task_loss_key]


def prettify(rel_error, is_percentage=True):
    if is_percentage:
        return f"{rel_error * 100:+.1f}%"
    else:
        return f"{rel_error:.2f}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--key", type=str, default="", help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument(
        "--num_to_avg", type=int, default=1, help="Number of final ckpts to average (for final loss fitting)"
    )
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    args = parser.parse_args()

    args.keys = KEYS_BY_KEY[args.key]

    return args


def get_final_configs(config_path: str):
    with open(config_path) as f:
        configs = json.load(f)
        configs = {name: FinalConfig(**config) for name, config in configs.items()}
    return configs


def get_data_by_name(configs: Dict[str, ExtrapolateNConfig], keys: List[str], min_step: Optional[int] = None):
    data_by_name: Dict = defaultdict(lambda: {"ns": [], "ds": [], "hs": [], "s1s": [], "s2s": [], "ys": []})
    for name, config in configs.items():
        n = config.n
        with open(config.path) as file_ref:
            reader = csv.DictReader(file_ref)
            lam = 0.999
            s1 = 0.0
            s2 = 0.0
            s2_momentum = 0
            last_lr = 0.0
            last_fake_lr = 0.0
            last_d = 0
            encountered_ds = set()
            for row in reader:
                d = int(float(row["throughput/total_tokens"]))
                if d in encountered_ds:
                    continue
                batch_size = int(row["batch_size_in_tokens"])
                steps = (d - last_d) / batch_size
                lr = float(row["optim/learning_rate_group0"])
                if lr > last_lr:  # warmup phase
                    fake_lr = float(row["learning_rate_peak"])
                    last_fake_lr = float(row["learning_rate_peak"])
                else:  # anneal phase
                    fake_lr = lr
                h = lr / float(row["learning_rate_peak"])
                s1 += fake_lr * steps
                s2_momentum = lam**steps * s2_momentum + (last_fake_lr - fake_lr) * steps
                s2 += s2_momentum
                last_lr = lr
                last_fake_lr = fake_lr
                last_d = d
                encountered_ds.add(d)
                y = np.average(
                    [float(row[key]) for key in keys], weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in keys]
                )
                if min_step is not None and d < min_step * batch_size:
                    continue
                data_by_name[name]["ns"].append(n)
                data_by_name[name]["ds"].append(d)
                data_by_name[name]["hs"].append(h)
                data_by_name[name]["s1s"].append(s1)
                data_by_name[name]["s2s"].append(s2)
                data_by_name[name]["ys"].append(y)
    return data_by_name


def get_final_data_by_name(configs, keys, num_to_avg=1):
    data_by_name: Dict = defaultdict(lambda: {"ns": [], "ds": [], "ys": []})
    for name, config in configs.items():
        n = config.n
        for path in config.paths:
            with open(path) as file_ref:
                reader = csv.DictReader(file_ref)
                rows = [row for row in reader]
                rows = rows[-num_to_avg:]
                ds, ys = [], []
                for row in rows:
                    d = int(float(row["throughput/total_tokens"]))
                    y = np.average(
                        [float(row[key]) for key in keys], weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in keys]
                    )
                    ds.append(d)
                    ys.append(y)
                d = np.mean(ds)
                y = np.mean(ys)
                data_by_name[name]["ns"].append(n)
                data_by_name[name]["ds"].append(d)
                data_by_name[name]["ys"].append(y)
        data_by_name[name]["mode"] = config.mode
    return data_by_name


def get_flops_data_by_name(configs, keys, num_to_avg=1):
    data_by_name: Dict = defaultdict(lambda: {"fs": [], "ys": []})
    for name, config in configs.items():
        for path in config.paths:
            with open(path) as file_ref:
                reader = csv.DictReader(file_ref)
                rows = [row for row in reader]
                rows = rows[-num_to_avg:]
                fs, ys = [], []
                for row in rows:
                    d = int(float(row["throughput/total_tokens"]))
                    f = d * MODEL_FLOPS[name]
                    y = np.average(
                        [float(row[key]) for key in keys], weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in keys]
                    )
                    fs.append(f)
                    ys.append(y)
                f = np.mean(fs)
                y = np.mean(ys)
                data_by_name[name]["fs"].append(f)
                data_by_name[name]["ys"].append(y)
    return data_by_name


def moving_average(arr, n):
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concat([ret[: n - 1] / np.arange(1, n), ret[n - 1 :] / n])


def get_length(path):
    return path.split("/")[-1].split(".csv")[0].split("-")[1]


def get_downstream_data_by_name(configs, keys, moving_avg=1, skip_perc=0.0, last_n_points=-1):
    loss_keys = tasks[keys].get_loss_keys()
    accuracy_keys = tasks[keys].get_accuracy_keys()

    data_by_name: Dict = defaultdict(lambda: {"xs": [], "ys": [], "ds": [], "ns": [], "ls": []})

    for name, config in configs.items():
        n = config.n
        for path in config.paths:
            length = get_length(path)
            with open(path) as file_ref:
                reader = csv.DictReader(file_ref)
                rows = [row for row in reader]
                xs, ys, ds, ns, ls = [], [], [], [], []
                for row in rows:
                    d = int(float(row["throughput/total_tokens"]))
                    x = np.average(
                        [float(row[key]) for key in loss_keys],
                        weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in loss_keys],
                    )
                    y = np.average(
                        [float(row[key]) for key in accuracy_keys],
                        weights=[WEIGHT_BY_KEY.get(key, 1.0) for key in accuracy_keys],
                    )
                    xs.append(x)
                    ys.append(y)
                    ds.append(d)
                    ns.append(n)
                    ls.append(length)

                if config.mode == "train":
                    # skip initial ckpts

                    xs = xs[int(np.ceil(skip_perc * len(xs))) :]
                    ys = ys[int(np.ceil(skip_perc * len(ys))) :]
                    ds = ds[int(np.ceil(skip_perc * len(ds))) :]
                    ns = ns[int(np.ceil(skip_perc * len(ns))) :]
                    ls = ls[int(np.ceil(skip_perc * len(ls))) :]

                    # apply moving_avg
                    xs = moving_average(xs, n=moving_avg).tolist()
                    # ys = ys[moving_avg-1:]
                    # ds = ds[moving_avg-1:]
                    # ns = ns[moving_avg-1:]
                    # ls = ls[moving_avg-1:]

                    # last n points
                    if last_n_points > 0:
                        xs = xs[-last_n_points:]
                        ys = ys[-last_n_points:]
                        ds = ds[-last_n_points:]
                        ns = ns[-last_n_points:]
                        ls = ls[-last_n_points:]

                data_by_name[name]["xs"] += xs
                data_by_name[name]["ys"] += ys
                data_by_name[name]["ds"] += ds
                data_by_name[name]["ns"] += ns
                data_by_name[name]["ls"] += ls

    return data_by_name


def get_ax(name):
    if "1xC" in name:
        return 0
    if "2xC" in name:
        return 1
    if "5xC" in name:
        return 2
    if "10xC" in name:
        return 3
    return 4
