import argparse
import json
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    get_final_configs,
    get_data_by_name,
    get_step1_data_by_name,
    get_task_sets,
    parse_args,
    prettify,
    tasks,
)

MARKERS = {"0.5xC": "D", "1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}
FONTSIZE = 9

D_START_STEP_BY_NAME = {
    "190m-0.5xC": 72625 * 0.9 * 0.05,
    "190m-1xC": 72625 * 0.9 * 0.1,
    "190m-2xC": 72625 * 0.9 * 0.2,
    "190m-5xC": 72625 * 0.9 * 0.5,
    "190m-10xC": 72625 * 0.9 * 1.0,
    "370m-0.5xC": 94427 * 0.9 * 0.05,
    "370m-1xC": 94427 * 0.9 * 0.1,
    "370m-2xC": 94427 * 0.9 * 0.2,
    "370m-5xC": 94427 * 0.9 * 0.5,
    "370m-10xC": 94427 * 0.9 * 1.0,
    "760m-0.5xC": 115706 * 0.9 * 0.05,
    "760m-1xC": 115706 * 0.9 * 0.1,
    "760m-2xC": 115706 * 0.9 * 0.2,
    "760m-5xC": 115706 * 0.9 * 0.5,
    "760m-10xC": 115706 * 0.9 * 1.0,
    "1.3b-0.5xC": 162694 * 0.9 * 0.05,
    "1.3b-1xC": 162694 * 0.9 * 0.1,
    "1.3b-2xC": 162694 * 0.9 * 0.2,
    "1.3b-5xC": 162694 * 0.9 * 0.5,
    "1.3b-10xC": 162694 * 0.9 * 1.0,
    "3.2b-0.5xC": 201524 * 0.9 * 0.05,
    "3.2b-1xC": 201524 * 0.9 * 0.1,
    "3.2b-2xC": 201524 * 0.9 * 0.2,
    "3.2b-5xC": 201524 * 0.9 * 0.5,
    "3.2b-10xC": 201524 * 0.9 * 1.0,
}


def main():
    args = parse_args()

    data_by_name = {}
    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}
        for name, config in configs.items():
            data_by_name = {
                **data_by_name,
                **get_data_by_name({name: config}, args.keys, min_step=int(D_START_STEP_BY_NAME[name])),
            }

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    for name, data in data_by_name.items():
        config = configs[name]
        x = data["ds"][-1]
        y = data['ys'][0] - data['ys'][-1]
        if "-10xC" in config.label:
            label = config.label[:-5]
        else:
            label = None
        ax.scatter(x, y, color="white", edgecolors=config.color, label=label, s=10)

    ax.set_xscale("log")
    ax.set_xlabel("Tokens (D)")
    ax.set_ylabel(f"Delta of loss, {args.key}")
    ax.legend(loc="lower left", fontsize=FONTSIZE)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
