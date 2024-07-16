import csv
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from .utils import get_coefficients


@dataclass
class ExtrapolateNConfig:
    path: str
    """
    Path containing the W&B downloaded data and metadata.
    """

    keys: List[str]
    """
    The metrics for computing the scaling law predictions.
    """

    mode: str
    """
    Whether this model is used for fitting the curve ('train') or evaluating the fit ('eval').
    """

    label: str
    """
    A short label for this curve.
    """

    color: str
    """
    The color for this curve.
    """


def get_data_at_d(config_by_n: Dict[int, ExtrapolateNConfig], d: int):
    """
    d: If its value is string "last", then loss from the last ckpt is used.
       If its value is an integer, then loss from the first ckpt with at least d tokens is used.
    """
    train_ns, train_ys, eval_ns, eval_ys = [], [], [], []
    for n, config in config_by_n.items():
        with open(config.path) as file_ref:
            reader = csv.DictReader(file_ref)
            y = None
            for row in reader:
                dd = int(float(row['throughput/total_tokens']))
                yy = np.mean([float(row[key]) for key in config.keys])
                if dd >= d and y is None:
                    y = yy
            if y is None: # there is no data at or later than d tokens
                continue
            if config.mode == 'train':
                train_ns.append(n)
                train_ys.append(y)
            elif config.mode == 'eval':
                eval_ns.append(n)
                eval_ys.append(y)
    return train_ns, train_ys, eval_ns, eval_ys


def plot_n_scaling_at_d(train_ns, train_ys, eval_ns, fitting_func, p0=[20, -0.1, 0.0], **plot_kwargs):
    coefficients = get_coefficients(train_ns, train_ys, fitting_func, p0=p0)

    plot_ns = np.linspace(0.8 * min(train_ns + eval_ns), 1.2 * max(train_ns + eval_ns), 1000)

    plt.plot(
        plot_ns,
        fitting_func(np.array(plot_ns), *coefficients),
        **plot_kwargs,
    )


def get_data_forall_d(config_by_n: Dict[int, ExtrapolateNConfig]):
    data_by_d = defaultdict(lambda: {'train_ns': [], 'train_ys': [], 'eval_ns': [], 'eval_ys': []})
    data_by_n = defaultdict(lambda: {'ds': [], 'ys': []})
    for n, config in config_by_n.items():
        with open(config.path) as file_ref:
            reader = csv.DictReader(file_ref)
            for row in reader:
                d = int(float(row['throughput/total_tokens']))
                y = np.mean([float(row[key]) for key in config.keys])
                if config.mode == 'train':
                    data_by_d[d]['train_ns'].append(n)
                    data_by_d[d]['train_ys'].append(y)
                elif config.mode == 'eval':
                    data_by_d[d]['eval_ns'].append(n)
                    data_by_d[d]['eval_ys'].append(y)
                data_by_n[n]['ds'].append(d)
                data_by_n[n]['ys'].append(y)
    return data_by_d, data_by_n


def plot_n_scaling_forall_d(data_by_d, data_by_n, config_by_n, fitting_func, p0=[20, -0.1, 0.0], **plot_kwargs):
    for n, data in data_by_n.items():
        config = config_by_n[n]
        plt.plot(data['ds'], data['ys'], color=config.color, linestyle='-', label=config.label, **plot_kwargs)

    predicted_data_by_n = defaultdict(lambda: {'ds': [], 'ys': []})
    for d, data in data_by_d.items():
        train_ns, train_ys, eval_ns = data['train_ns'], data['train_ys'], data['eval_ns']
        if len(train_ns) < 3:
            continue
        coefficients = get_coefficients(train_ns, train_ys, fitting_func, p0=p0)
        for n in eval_ns:
            predicted_data_by_n[n]['ds'].append(d)
            predicted_data_by_n[n]['ys'].append(fitting_func(n, *coefficients))

    for n, data in predicted_data_by_n.items():
        config = config_by_n[n]
        plt.plot(data['ds'], data['ys'], color=config.color, linestyle='--', label=f'{config.label} (predicted)', **plot_kwargs)
