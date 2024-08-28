import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import scipy

from .utils import get_coefficients_huber


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


def get_config_by_n(configs: Dict[str, ExtrapolateNConfig], n: int):
    for config in configs.values():
        if config.n == n:
            return config
    raise ValueError(f"Could not find config for n={n}")


def get_data_forall_n(configs: Dict[str, ExtrapolateNConfig]):
    data_by_n = defaultdict(lambda: {'ds': [], 'ys': []})
    for name, config in configs.items():
        n = config.n
        with open(config.path) as file_ref:
            reader = csv.DictReader(file_ref)
            for row in reader:
                d = int(float(row['throughput/total_tokens']))
                y = np.mean([float(row[key]) for key in config.keys])
                data_by_n[n]['ds'].append(d)
                data_by_n[n]['ys'].append(y)
    return data_by_n


def plot_n_d_scaling(data_by_n, configs, fitting_func, grad_func, p0, bounds, **plot_kwargs):
    # fit the parameters
    train_nds, train_ys = [], []
    for n, data in data_by_n.items():
        config = get_config_by_n(configs, n)
        if config.mode == 'train':
            train_nds += [[n, d] for d in data['ds']]
            train_ys += data['ys']
            # DMAX = 104857600000 * 2
            # train_nds += [[n, d] for d in data['ds'] if d <= DMAX]
            # train_ys += [y for d, y in zip(data['ds'], data['ys']) if d <= DMAX]
    coefficients = get_coefficients_huber(train_nds, train_ys, fitting_func, grad_func, p0=p0, bounds=bounds)
    predicted_data_by_n = {}
    for n, data in data_by_n.items():
        predicted_data_by_n[n] = {
            'ds': data['ds'],
            'ys': [fitting_func([n, d], coefficients) for d in data['ds']],
        }

    # plot the actual data
    for n, data in data_by_n.items():
        config = get_config_by_n(configs, n)
        plt.scatter(data['ds'], data['ys'], color='white', edgecolors=config.color, label=config.label, s=5.0, **plot_kwargs)

    # plot the fitted curve
    for n, data in predicted_data_by_n.items():
        config = get_config_by_n(configs, n)
        if config.mode == 'train':
            plt.plot(data['ds'], data['ys'], color=config.color, linestyle='--', linewidth=0.8, label=f'{config.label} (fitted)', **plot_kwargs)
        else:
            plt.plot(data['ds'], data['ys'], color=config.color, linestyle='--', linewidth=0.8, label=f'{config.label} (predicted)', **plot_kwargs)

    # # plot the residue
    # for n in data_by_n:
    #     config = get_config_by_n(configs, n)
    #     plt.scatter(
    #         data_by_n[n]['ds'],
    #         np.array(data_by_n[n]['ys']) - np.array(predicted_data_by_n[n]['ys']),
    #         color='white',
    #         edgecolors=config.color,
    #         label=config.label,
    #         s=5.0,
    #         **plot_kwargs,
    #     )

    # # fit the residue
    # ns, rs = [], []
    # for n in data_by_n:
    #     r = predicted_data_by_n[n]['ys'][-1] - data_by_n[n]['ys'][-1]
    #     ns.append(n)
    #     rs.append(r)
    # plt.scatter(ns, rs)
    # fun = lambda x, a, b, c : a * np.log(x + b) + c
    # coeffs = scipy.optimize.curve_fit(fun, ns, rs, p0=[1.0, 0.0, 0.0], maxfev=50000)[0]
    # xs = np.linspace(0, max(ns), 100)
    # ys = [fun(x, *coeffs) for x in xs]
    # plt.plot(xs, ys, color='black', linestyle='-', linewidth=0.8, label='log fit', **plot_kwargs)
    # print(coeffs)
