import csv
from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy

from .curve_fit import get_coefficients


@dataclass
class CurveFitConfig:
    path: str
    """
    Path containing the W&B downloaded data and metadata.
    """

    keys: List[str]
    """
    The metrics for computing the scaling law predictions.
    """

    n: int
    """
    The number of parameters in the model.
    """

    mode: str
    """
    Whether this model is used for fitting the curve ('train') or evaluating the fit ('eval').
    """


def get_data_at_d(config_by_n: Dict[int, CurveFitConfig], d: int):
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
