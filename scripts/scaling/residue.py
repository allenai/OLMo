import json

import matplotlib.pyplot as plt
import numpy as np

from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    chinchilla_n_d_lr_power_fit,
    get_ax,
    get_coefficients_huber,
    get_data_by_name,
    grad_chinchilla_n_d_lr_power_fit,
    parse_args,
)


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}

    data_by_name = get_data_by_name(configs, args.keys, min_step=3000)
    const_configs = {
        name: ExtrapolateNConfig(
            path=config.path.replace('5shot', 'const').replace('1xC', '10xC').replace('2xC', '10xC').replace('5xC', '10xC'),
            mode=config.mode, n=config.n, label=config.label, color=config.color)
        for name, config in configs.items()
    }
    const_data_by_name = get_data_by_name(const_configs, args.keys, min_step=3000)

    num_axs = 5
    fig, axs = plt.subplots(1, num_axs, figsize=(num_axs * 8, 6))

    for name in data_by_name:
        config = configs[name]
        data = data_by_name[name]
        const_data = const_data_by_name[name]
        if not data['ds'][:-1] == const_data['ds'][:len(data['ds'])-1]:
            print(name)
            print(data['ds'][:-1])
            print(const_data['ds'][:len(data['ds'])-1])
        # assert data['ds'][:-1] == const_data['ds'][:len(data['ds'])-1]
        ds = data['ds'][:-2]
        residues = np.array(data['ys'][:-2]) - np.array(const_data['ys'][:len(data['ys'])-2])

        ax = axs[get_ax(name)]
        ax.scatter(
            ds,
            residues,
            color='white',
            edgecolors=config.color,
            label=config.label,
            s=5.0,
        )

    for ax in axs:
        ax.legend(loc="upper right", ncols=2, fontsize=10)
        ax.set_xlabel("Tokens (d)")
    axs[0].set_ylabel(f"Residue of CE loss, {args.key if args.key != '' else args.keys}")
    plt.suptitle("Residue against curve of const LR schedule")
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
