import json
import matplotlib.pyplot as plt
import numpy as np
from olmo.scaling.scaling_laws.utils import (
    parse_args,
    ExtrapolateNConfig, get_config_by_n, get_data_forall_n,
    chinchilla_n_d_fit, grad_chinchilla_n_d_fit,
    get_coefficients_huber,
)


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}

    data_by_n = get_data_forall_n(configs, args.keys, final_only=args.final_only)

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
        p0=[7.0, 11.5, 0.5, 0.5, 2.0],
        bounds=[(0, None), (0, None), (0, None), (0, None), (0, None)],
    )
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)

    # make predictions
    predicted_data_by_n = {}
    plotted_predicted_data_by_n = {}

    if args.final_only:
        plot_ds = []
        predicted_data = []
        plotted_predicted_data = []
        for n, data in data_by_n.items():
            plot_ds.append(data['ds'])

        predicted_data = {
            'ds': plot_ds,
            'ys': [chinchilla_n_d_fit([n, d], coefficients) for d in plot_ds],
        }
        plot_ds = np.linspace(min(plot_ds), max(plot_ds), 100) 
        plotted_predicted_data = {
            'ds': plot_ds,
            'ys': [chinchilla_n_d_fit([n, d], coefficients) for d in plot_ds],
        }


    for n, data in data_by_n.items():
        ds = data['ds']
        predicted_data_by_n[n] = {
            'ds': ds,
            'ys': [chinchilla_n_d_fit([n, d], coefficients) for d in ds],
        }
        ds = np.linspace(min(data['ds']), max(data['ds']), 100)
        plotted_predicted_data_by_n[n] = {
            'ds': ds,
            'ys': [chinchilla_n_d_fit([n, d], coefficients) for d in ds],
        }

    # plot the actual data
    for n, data in data_by_n.items():
        config = get_config_by_n(configs, n)
        if args.final_only:
            plt.scatter(data['ds'], data['ys'], color='white', edgecolors=config.color, s=5.0)
        else:
            plt.scatter(data['ds'], data['ys'], color='white', edgecolors=config.color, label=config.label, s=5.0)

        predicted_data = predicted_data_by_n[n]
        for d, y, y_pred in zip(data['ds'], data['ys'], predicted_data['ys']):
            rel_error = (y_pred - y) / y
            plt.annotate(f'{config.label}: {rel_error * 100:+.1f}%', (d, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6, color=config.color)

    # plot the fitted curve
    if args.final_only:
        plt.plot(plotted_predicted_data['ds'], plotted_predicted_data['ys'], color="black", linestyle='--', linewidth=0.8)
    else:
        for n, data in plotted_predicted_data_by_n.items():
            config = get_config_by_n(configs, n)
            if config.mode == 'train':
                plt.plot(data['ds'], data['ys'], color=config.color, linestyle='--', linewidth=0.8, label=f'{config.label} (fitted)')
            else:
                plt.plot(data['ds'], data['ys'], color=config.color, linestyle='--', linewidth=0.8, label=f'{config.label} (predicted)')
    plt.text(
        x=0.25, y=0.50,
        s=f"L(n, d) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )

    plt.legend(loc="upper right", ncols=2)
    plt.xlabel("Tokens (d)")
    plt.ylabel(f"CE loss, {args.key if args.key != '' else args.keys}")
    plt.title(f"Fitting final loss")
    plt.savefig(args.output_path, dpi=300)

    y_7b_2T = chinchilla_n_d_fit([6682316800, 2e12], coefficients)
    print(f'Predicted final loss for 7b-2T: {y_7b_2T:.2f}')
    y_7b_3T = chinchilla_n_d_fit([6682316800, 3e12], coefficients)
    print(f'Predicted final loss for 7b-3T: {y_7b_3T:.2f}')
    y_13b_5T = chinchilla_n_d_fit([13e9, 5e12], coefficients)
    print(f'Predicted final loss for 13b-5T: {y_13b_5T:.2f}')


if __name__ == "__main__":
    main()
