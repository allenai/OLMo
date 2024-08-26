import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import numpy as np
from olmo.scaling.scaling_laws.joint import (
    get_config_by_n,
    get_data_forall_n,
)
from olmo.scaling.scaling_laws.utils import (
    validation, downstream_bpb,
    chinchilla_n_d_fit, grad_chinchilla_n_d_fit,
    get_coefficients_huber,
)


@dataclass
class Config:
    path: str
    mode: str
    n: int
    label: str
    color: str


def get_config_by_n(configs, n):
    for config in configs.values():
        if config.n == n:
            return config
    raise ValueError(f"Could not find config for n={n}")


def get_data_forall_n(configs, keys):
    data_by_n = defaultdict(lambda: {'ds': [], 'ys': []})
    for name, config in configs.items():
        n = config.n
        with open(config.path) as file_ref:
            reader = csv.DictReader(file_ref)
            for row in reader:
                d = int(float(row['throughput/total_tokens']))
                y = np.mean([float(row[key]) for key in keys])
                data_by_n[n]['ds'].append(d)
                data_by_n[n]['ys'].append(y)
    return data_by_n


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", type=str, default="", help="For avg metrics. Use one of [all-val-lm, all-bpb]")
    parser.add_argument("--keys", nargs='+', type=str, help="For individual metrics")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    args = parser.parse_args()

    if args.key == 'all-val-lm':
        args.keys = [f'eval/{val}/CrossEntropyLoss' for val in validation]
    elif args.key == 'all-bpb':
        args.keys = [f'eval/downstream_bpb/{task}_bpb' for task in downstream_bpb]

    return args


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: Config(**config) for name, config in configs.items()}

    data_by_n = get_data_forall_n(configs, args.keys)

    plt.figure()

    train_nds, train_ys = [], []
    for n, data in data_by_n.items():
        config = get_config_by_n(configs, n)
        if config.mode == 'train':
            train_nds += [[n, d] for d in data['ds']]
            train_ys += data['ys']

    # fit the parameters
    coefficients = get_coefficients_huber(
        train_nds, train_ys,
        chinchilla_n_d_fit, grad_chinchilla_n_d_fit,
        p0=[1.5, 2.5, 0.5, 0.5, 2.0],
        bounds=[(0, None), (0, None), (0, None), (0, None), (0, None)],
    )
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)

    # make predictions
    predicted_data_by_n = {}
    for n, data in data_by_n.items():
        ds = np.linspace(min(data['ds']), max(data['ds']), 100)
        predicted_data_by_n[n] = {
            'ds': ds,
            'ys': [chinchilla_n_d_fit([n, d], coefficients) for d in ds],
        }

    # plot the actual data
    for n, data in data_by_n.items():
        config = get_config_by_n(configs, n)
        plt.scatter(data['ds'], data['ys'], color='white', edgecolors=config.color, label=config.label, s=5.0)

    # plot the fitted curve
    for n, data in predicted_data_by_n.items():
        config = get_config_by_n(configs, n)
        if config.mode == 'train':
            plt.plot(data['ds'], data['ys'], color=config.color, linestyle='--', linewidth=0.8, label=f'{config.label} (fitted)')
        else:
            plt.plot(data['ds'], data['ys'], color=config.color, linestyle='--', linewidth=0.8, label=f'{config.label} (predicted)')
    plt.text(
        x=0.30, y=0.50,
        s=f"L(n, d) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )

    plt.legend(loc="upper right", ncols=2)
    plt.xlabel("Tokens (d)")
    plt.ylabel(f"CE loss, {args.key if args.key != '' else args.keys}")
    plt.title(f"Fitting final loss")
    plt.savefig(args.output_path, dpi=300)


if __name__ == "__main__":
    main()
