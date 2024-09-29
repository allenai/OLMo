import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy

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

downstream = [
    "hellaswag_len_norm",
    "hellaswag_rc_0shot_len_norm",
    "hellaswag_rc_5shot_len_norm",
    "hellaswag_mc_5shot_acc",
    "winogrande_acc",
    "winogrande_rc_0shot_acc",
    "winogrande_rc_5shot_acc",
    "winogrande_mc_5shot_acc",
    "piqa_len_norm",
    "piqa_rc_0shot_len_norm",
    "piqa_rc_5shot_len_norm",
    "piqa_mc_5shot_acc",
    "social_iqa_len_norm",
    "socialiqa_rc_0shot_len_norm",
    "socialiqa_rc_5shot_len_norm",
    "socialiqa_mc_5shot_acc",
    "openbook_qa_len_norm",
    "openbookqa_rc_0shot_len_norm",
    "openbookqa_rc_5shot_len_norm",
    "openbookqa_mc_5shot_acc",
    "commonsense_qa_len_norm",
    "csqa_rc_0shot_len_norm",
    "csqa_rc_5shot_len_norm",
    "csqa_mc_5shot_acc",
    "boolq_acc",
    "boolq_rc_0shot_acc",
    "boolq_rc_5shot_acc",
    "boolq_mc_5shot_acc",
    "copa_acc",
    "copa_rc_0shot_acc",
    "arc_easy_acc",
    "arc_easy_rc_0shot_acc",
    "arc_easy_rc_5shot_acc",
    "arc_easy_mc_5shot_acc",
    "arc_challenge_len_norm",
    "arc_challenge_rc_0shot_len_norm",
    "arc_challenge_rc_5shot_len_norm",
    "arc_challenge_mc_5shot_acc",
    "sciq_acc",
    "sciq_rc_0shot_acc",
    "mmlu_social_sciences_var_len_norm",
    "mmlu_humanities_var_len_norm",
    "mmlu_other_var_len_norm",
    "mmlu_stem_mc_5shot_test_len_norm",
    "mmlu_humanities_mc_5shot_len_norm",
    "mmlu_social_sciences_mc_5shot_len_norm",
    "mmlu_stem_var_len_norm",
    "mmlu_other_mc_5shot_test_len_norm",
    "mmlu_humanities_mc_5shot_test_len_norm",
    "mmlu_stem_mc_5shot_len_norm",
    "mmlu_social_sciences_mc_5shot_test_len_norm",
    "mmlu_other_mc_5shot_len_norm",
    "basic_arithmetic_acc",
]


downstream_bpb = [
    "piqa_rc_0shot_bpb",
    "piqa_rc_5shot_bpb",
    "piqa_mc_5shot_bpb",
    "hellaswag_rc_0shot_bpb",
    "hellaswag_rc_5shot_bpb",
    "hellaswag_mc_5shot_bpb",
    "winogrande_rc_0shot_bpb",
    "winogrande_rc_5shot_bpb",
    "winogrande_mc_5shot_bpb",
    "openbookqa_rc_0shot_bpb",
    "openbookqa_rc_5shot_bpb",
    "openbookqa_mc_5shot_bpb",
    "boolq_rc_0shot_bpb",
    "boolq_rc_5shot_bpb",
    "boolq_mc_5shot_bpb",
    "sciq_rc_0shot_bpb",
    # "sciq_rc_5shot_bpb",
    # "sciq_mc_5shot_bpb",
    "arc_easy_rc_0shot_bpb",
    "arc_easy_rc_5shot_bpb",
    "arc_easy_mc_5shot_bpb",
    "arc_challenge_rc_0shot_bpb",
    "arc_challenge_rc_5shot_bpb",
    "arc_challenge_mc_5shot_bpb",
    "copa_rc_0shot_bpb",
    # "copa_rc_5shot_bpb",
    # "copa_mc_5shot_bpb",
    "csqa_rc_0shot_bpb",
    "csqa_rc_5shot_bpb",
    "csqa_mc_5shot_bpb",
    "socialiqa_rc_0shot_bpb",
    "socialiqa_rc_5shot_bpb",
    "socialiqa_mc_5shot_bpb",
    "mmlu_stem_var_bpb",
    "mmlu_humanities_var_bpb",
    "mmlu_social_sciences_var_bpb",
    "mmlu_other_var_bpb",
    "mmlu_stem_bpb",
    "mmlu_humanities_bpb",
    "mmlu_social_sciences_bpb",
    "mmlu_other_bpb",
]


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


KEYS_BY_KEY = {
    "all-val-lm": [f"eval/{val}/CrossEntropyLoss" for val in validation],
    "all-bpb": [f"eval/downstream_bpb/{task}_bpb" for task in downstream_bpb],
    "c4": ["eval/c4_en-validation/CrossEntropyLoss"],
    "mmlu": [
        f"eval/downstream_bpb/{task}_bpb"
        for task in [
            "mmlu_stem_var_bpb",
            "mmlu_humanities_var_bpb",
            "mmlu_social_sciences_var_bpb",
            "mmlu_other_var_bpb",
        ]
    ],
    "hellaswag-5shot": ["eval/downstream_bpb/hellaswag_rc_5shot_bpb_bpb"],
    "arc-e-5shot": ["eval/downstream_bpb/arc_easy_rc_5shot_bpb_bpb"],
    "arc-c-5shot": ["eval/downstream_bpb/arc_challenge_rc_5shot_bpb_bpb"],
    "piqa-5shot": ["eval/downstream_bpb/piqa_rc_5shot_bpb_bpb"],
    "winogrande-5shot": ["eval/downstream_bpb/winogrande_rc_5shot_bpb_bpb"],
    "openbookqa-5shot": ["eval/downstream_bpb/openbookqa_rc_5shot_bpb_bpb"],
    "boolq-5shot": ["eval/downstream_bpb/boolq_rc_5shot_bpb_bpb"],
    "sciq-0shot": ["eval/downstream_bpb/sciq_rc_0shot_bpb_bpb"],
    "copa-0shot": ["eval/downstream_bpb/copa_rc_0shot_bpb_bpb"],
    "csqa-5shot": ["eval/downstream_bpb/csqa_rc_5shot_bpb_bpb"],
    "socialiqa-5shot": ["eval/downstream_bpb/socialiqa_rc_5shot_bpb_bpb"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--key", type=str, default="", help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument("--num_to_avg", type=int, default=1, help="Number of final ckpts to average (for final loss fitting)")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    args = parser.parse_args()

    args.keys = KEYS_BY_KEY[args.key]

    return args


def get_data_by_name(configs, keys, min_step=None):
    data_by_name = defaultdict(lambda: {"ns": [], "ds": [], "hs": [], "s1s": [], "s2s": [], "ys": []})
    for name, config in configs.items():
        n = config.n
        with open(config.path) as file_ref:
            reader = csv.DictReader(file_ref)
            lam = 0.999
            s1 = 0
            s2 = 0
            s2_momentum = 0
            last_lr = 0
            last_fake_lr = 0
            last_d = 0
            encountered_ds = set()
            for row in reader:
                d = int(float(row["throughput/total_tokens"]))
                if d in encountered_ds:
                    continue
                batch_size = int(row["batch_size_in_tokens"])
                steps = (d - last_d) / batch_size
                lr = float(row["optim/learning_rate_group0"])
                if lr > last_lr: # warmup phase
                    fake_lr = float(row["learning_rate_peak"])
                    last_fake_lr = float(row["learning_rate_peak"])
                else: # anneal phase
                    fake_lr = lr
                h = lr / float(row["learning_rate_peak"])
                s1 += fake_lr * steps
                s2_momentum = lam**steps * s2_momentum + (last_fake_lr - fake_lr) * steps
                s2 += s2_momentum
                last_lr = lr
                last_fake_lr = fake_lr
                last_d = d
                encountered_ds.add(d)
                y = np.mean([float(row[key]) for key in keys])
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
    data_by_name = defaultdict(lambda: {"ns": [], "ds": [], "ys": []})
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
                    y = np.mean([float(row[key]) for key in keys])
                    ds.append(d)
                    ys.append(y)
                d = np.mean(ds)
                y = np.mean(ys)
                data_by_name[name]["ns"].append(n)
                data_by_name[name]["ds"].append(d)
                data_by_name[name]["ys"].append(y)
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


# Power Law functions


def openai_fit(x, a, b, c):
    return (a / x + c) ** b


def chinchilla_fit(x, a, b, c):
    return a * x**b + c


def chinchilla_contaminated_fit(x, a, b, c, d):
    return (a * x**b + c) * (1 - x / d)


# Scipy curve_fit with least squares
def get_coefficients(train_xs, train_ys, fitting_func, p0):
    if isinstance(train_xs[0], list):
        train_xs = np.array(train_xs).transpose()
    coeffs = scipy.optimize.curve_fit(fitting_func, train_xs, train_ys, p0=p0, maxfev=50000)[0]
    coeffs_string = ", ".join([chr(ord("a") + i) + f" = {coeffs[i]:.2f}" for i in range(len(coeffs))])
    print(f"{fitting_func.__name__}: {coeffs_string}")
    return coeffs


# x = flops
# p[0] = A, p[1] = B, p[2] = E
def chinchilla_flops_fit(x, p):
    # return ax**b + E
    return p[0] * x ** p[1] + p[2]


def grad_chinchilla_flops_fit(x, p):
    grad_A = x ** p[1]
    grad_B = p[0] * x ** p[1] * np.log(x)
    grad_E = 1
    return [grad_A, grad_B, grad_E]


# x[0] = d, x[1] = h
# p[0] = b = log100(B), p[1] = beta, p[2] = E, p[3] = F
def chinchilla_d_lr_fit(x, p):
    return 100 ** p[0] / x[0] ** p[1] + p[2] + p[3] * x[1]


def grad_chinchilla_d_lr_fit(x, p):
    grad_b = (1 / x[0] ** p[1]) * (100 ** p[0] * np.log(100))
    grad_beta = -(100 ** p[0]) * np.log(x[0]) / x[0] ** p[1]
    grad_E = 1
    grad_F = x[1]
    return [grad_b, grad_beta, grad_E, grad_F]


# x[0] = n, x[1] = d
# p[0] = a = log(A), p[1] = b = log(B), p[2] = alpha, p[3] = beta, p[4] = E
def chinchilla_n_d_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4]


def grad_chinchilla_n_d_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E]


# x[0] = n, x[1] = d, x[2] = h
# p[0] = a = log(A), p[1] = b = log(B), p[2] = alpha, p[3] = beta, p[4] = E, p[5] = F
def chinchilla_n_d_lr_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E + F * x[2]
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] + p[5] * x[2]


def grad_chinchilla_n_d_lr_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = x[2]
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F]


def chinchilla_n_d_lr_log_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E + F * x[2] * np.log(x[0] / e**r + s)
    return (
        np.exp(p[0]) / x[0] ** p[2]
        + np.exp(p[1]) / x[1] ** p[3]
        + p[4]
        + p[5] * x[2] * np.log(x[0] / np.exp(p[6]) + p[7])
    )


def grad_chinchilla_n_d_lr_log_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = x[2] * np.log(x[0] / np.exp(p[6]) + p[7])
    grad_r = p[5] * x[2] * (1 / (x[0] / np.exp(p[6]) + p[7])) * x[0] * (-1 / np.exp(p[6]))
    grad_s = p[5] * x[2] * (1 / (x[0] / np.exp(p[6]) + p[7]))
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def chinchilla_n_d_lr_power_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E + F * x[2] * x[0]**r
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] + p[5] * x[2] * x[0] ** p[6]


def grad_chinchilla_n_d_lr_power_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = x[2] * x[0] ** p[6]
    grad_r = p[5] * x[2] * x[0] ** p[6] * np.log(x[0])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r]


def chinchilla_n_d_lr_power_minus_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * (1 - x[2]) * x[0] ** p[6]


def grad_chinchilla_n_d_lr_power_minus_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = - (1 - x[2]) * x[0] ** p[6]
    grad_r = - p[5] * (1 - x[2]) * x[0] ** p[6] * np.log(x[0])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r]


def chinchilla_n_d_lr_power_minus_powerd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r * x[1]**s
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * (1 - x[2]) * x[0] ** p[6] * x[1] ** p[7]


def grad_chinchilla_n_d_lr_power_minus_powerd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = - (1 - x[2]) * x[0] ** p[6] * x[1] ** p[7]
    grad_r = - p[5] * (1 - x[2]) * x[1] ** p[7] * x[0] ** p[6] * np.log(x[0])
    grad_s = - p[5] * (1 - x[2]) * x[0] ** p[6] * x[1] ** p[7] * np.log(x[1])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def chinchilla_n_d_lr_power_minus_powertd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r * (x[1]**s + t)
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * (1 - x[2]) * x[0] ** p[6] * (x[1] ** p[7] + np.exp(p[8]))


def grad_chinchilla_n_d_lr_power_minus_powertd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = - (1 - x[2]) * x[0] ** p[6] * (x[1] ** p[7] + np.exp(p[8]))
    grad_r = - p[5] * (1 - x[2]) * (x[1] ** p[7] + np.exp(p[8])) * x[0] ** p[6] * np.log(x[0])
    grad_s = - p[5] * (1 - x[2]) * x[0] ** p[6] * x[1] ** p[7] * np.log(x[1])
    grad_t = - p[5] * (1 - x[2]) * x[0] ** p[6] * np.exp(p[8])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s, grad_t]


def chinchilla_n_d_lr_power_minus_logtd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r * (log(x[1]) + s)
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * (1 - x[2]) * x[0] ** p[6] * (np.log(x[1]) + p[7])


def grad_chinchilla_n_d_lr_power_minus_logtd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = - (1 - x[2]) * x[0] ** p[6] * (np.log(x[1]) + p[7])
    grad_r = - p[5] * (1 - x[2]) * (np.log(x[1]) + p[7]) * x[0] ** p[6] * np.log(x[0])
    grad_s = - p[5] * (1 - x[2]) * x[0] ** p[6]
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def chinchilla_n_d_lr_logt_minus_logtd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) / (log(x[0]) + r) * (log(x[1]) + s)
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * (1 - x[2]) / (np.log(x[0]) + p[6]) * (np.log(x[1]) + p[7])


def grad_chinchilla_n_d_lr_logt_minus_logtd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = - (1 - x[2]) / (np.log(x[0]) + p[6]) * (np.log(x[1]) + p[7])
    grad_r = - p[5] * (1 - x[2]) * (np.log(x[1]) + p[7]) * (-1 / (np.log(x[0]) + p[6])**2)
    grad_s = - p[5] * (1 - x[2]) / (np.log(x[0]) + p[6])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def tissue_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * x[2] * x[0]**r
    return max(1e-8, np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * x[2] * x[0] ** p[6])


def grad_tissue_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -x[2] * x[0] ** p[6]
    grad_r = -p[5] * x[2] * x[0] ** p[6] * np.log(x[0])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r]


# Scipy minimize w/ Huber loss
def get_coefficients_huber(
    train_xs, train_ys, fitting_func, grad_func, p0, bounds, disp: bool = True, max_iter: int = 10000
):
    def huber_loss(x, delta):
        if np.abs(x) < delta:
            return 0.5 * x**2
        else:
            return delta * (np.abs(x) - 0.5 * delta)

    def loss_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        loss = np.sum(
            [huber_loss(np.log(pred) - np.log(actual), delta=delta) for actual, pred in zip(actuals, preds)]
        )
        return loss

    def jac_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        grads = [grad_func(x, p) for x in train_xs]
        us = [np.log(pred) - np.log(actual) for actual, pred in zip(actuals, preds)]
        grad_us = [u if np.abs(u) < delta else (delta * np.abs(u) / u) for u in us]
        results = [
            np.sum([grad_u * (1 / pred) * grad[i] for grad_u, pred, grad in zip(grad_us, preds, grads)])
            for i in range(len(grads[0]))
        ]
        return results

    assert len(train_xs) == len(train_ys)
    delta = 1e-3
    res = scipy.optimize.minimize(
        loss_fn,
        p0,
        args=(train_xs, train_ys, delta),
        jac=jac_fn,
        bounds=bounds,
        tol=0.0,
        method="L-BFGS-B",
        options={"ftol": 0.0, "gtol": 1e-10, "maxiter": max_iter, "disp": disp},
    )
    # res = scipy.optimize.minimize(loss_fn, p0, args=(train_xs, train_ys, delta), jac=jac_fn, tol=0.0, method='BFGS', options={'gtol': 1e-10, 'maxiter': 10000, 'disp': True})
    # print(res.message)
    coeffs = res.x
    if disp:
        print(f"coeffs: {coeffs}")
    return coeffs


def get_coefficients_huber_nolog(
    train_xs, train_ys, fitting_func, grad_func, p0, bounds, disp: bool = True, max_iter: int = 10000
):
    def huber_loss(x, delta):
        if np.abs(x) < delta:
            return 0.5 * x**2
        else:
            return delta * (np.abs(x) - 0.5 * delta)

    def loss_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        loss = np.sum(
            [huber_loss(pred - actual, delta=delta) for actual, pred in zip(actuals, preds)]
        )
        return loss

    def jac_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        grads = [grad_func(x, p) for x in train_xs]
        us = [pred - actual for actual, pred in zip(actuals, preds)]
        grad_us = [u if np.abs(u) < delta else (delta * np.abs(u) / u) for u in us]
        results = [
            np.sum([grad_u * grad[i] for grad_u, pred, grad in zip(grad_us, preds, grads)])
            for i in range(len(grads[0]))
        ]
        return results

    assert len(train_xs) == len(train_ys)
    delta = 1e-3
    res = scipy.optimize.minimize(
        loss_fn,
        p0,
        args=(train_xs, train_ys, delta),
        jac=jac_fn,
        bounds=bounds,
        tol=0.0,
        method="L-BFGS-B",
        options={"ftol": 0.0, "gtol": 1e-10, "maxiter": max_iter, "disp": disp},
    )
    # res = scipy.optimize.minimize(loss_fn, p0, args=(train_xs, train_ys, delta), jac=jac_fn, tol=0.0, method='BFGS', options={'gtol': 1e-10, 'maxiter': 10000, 'disp': True})
    # print(res.message)
    coeffs = res.x
    loss = res.fun
    if disp:
        print(f"coeffs: {coeffs}")
    return coeffs, loss