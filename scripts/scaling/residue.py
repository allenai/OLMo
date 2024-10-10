import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    get_ax,
    get_data_by_name,
    parse_args,
    get_coefficients_huber_nolog,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--key", type=str, default="", help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument("--keys", nargs="+", type=str, help="For individual metrics")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    parser.add_argument("--vfunc", type=str, default="")
    parser.add_argument("--ufunc", type=str, default="")
    args = parser.parse_args()

    if args.key == "all-val-lm":
        args.keys = [f"eval/{val}/CrossEntropyLoss" for val in validation]
    elif args.key == "all-bpb":
        args.keys = [f"eval/downstream_bpb/{task}_bpb" for task in downstream_bpb]
    elif args.key == "mmlu-var-bpb":
        args.keys = [
            f"eval/downstream_bpb/{task}_bpb"
            for task in [
                "mmlu_stem_var_bpb",
                "mmlu_humanities_var_bpb",
                "mmlu_social_sciences_var_bpb",
                "mmlu_other_var_bpb",
            ]
        ]

    return args


MARKER_BY_C = {
    1: 's',
    2: 'P',
    5: 'p',
    10: '*',
}

NS = [151898880, 319980544, 530074944, 681297408, 1176832000]

def func_pow_r(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4, r)
    return p[NS.index(x[0])] * x[1] ** p[5]
def jac_pow_r(x, p):
    grad = [0] * 6
    grad[NS.index(x[0])] = x[1] ** p[5]
    grad[5] = p[NS.index(x[0])] * x[1] ** p[5] * np.log(x[1])
    return grad

def func_pow_r_t(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4, r, t)
    return p[NS.index(x[0])] * (x[1] ** p[5] + p[6])
def jac_pow_r_t(x, p):
    grad = [0] * 7
    grad[NS.index(x[0])] = x[1] ** p[5] + p[6]
    grad[5] = p[NS.index(x[0])] * x[1] ** p[5] * np.log(x[1])
    grad[6] = p[NS.index(x[0])]
    return grad

def func_pow_r_splus_t(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4, r, s, t)
    return p[NS.index(x[0])] * ((x[1] + np.exp(p[6])) ** p[5] + p[7])
def jac_pow_r_splus_t(x, p):
    grad = [0] * 8
    grad[NS.index(x[0])] = (x[1] + np.exp(p[6])) ** p[5] + p[7]
    grad[5] = p[NS.index(x[0])] * (x[1] + np.exp(p[6])) ** p[5] * np.log(x[1] + np.exp(p[6]))
    grad[6] = p[NS.index(x[0])] * p[5] * (x[1] + np.exp(p[6])) ** (p[5] - 1) * np.exp(p[6])
    grad[7] = p[NS.index(x[0])]
    return grad

def func_pow_r_sminus_t(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4, r, s, t)
    return p[NS.index(x[0])] * ((x[1] - np.exp(p[6])) ** p[5] + p[7])
def jac_pow_r_sminus_t(x, p):
    grad = [0] * 8
    grad[NS.index(x[0])] = (x[1] - np.exp(p[6])) ** p[5] + p[7]
    grad[5] = p[NS.index(x[0])] * (x[1] - np.exp(p[6])) ** p[5] * np.log(x[1] - np.exp(p[6]))
    grad[6] = p[NS.index(x[0])] * p[5] * (x[1] - np.exp(p[6])) ** (p[5] - 1) * np.exp(p[6]) * -1
    grad[7] = p[NS.index(x[0])]
    return grad

def func_log(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4)
    return p[NS.index(x[0])] * np.log(x[1])
def jac_log(x, p):
    grad = [0] * 5
    grad[NS.index(x[0])] = np.log(x[1])
    return grad

def func_log_t(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4, t)
    return p[NS.index(x[0])] * (np.log(x[1]) + p[5])
def jac_log_t(x, p):
    grad = [0] * 6
    grad[NS.index(x[0])] = np.log(x[1]) + p[5]
    grad[5] = p[NS.index(x[0])]
    return grad

def func_log_splus_t(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4, s, t)
    return p[NS.index(x[0])] * (np.log(x[1] + np.exp(p[5])) + p[6])
def jac_log_splus_t(x, p):
    grad = [0] * 7
    grad[NS.index(x[0])] = np.log(x[1] + np.exp(p[5])) + p[6]
    grad[5] = p[NS.index(x[0])] * 1 / (x[1] + np.exp(p[5])) * np.exp(p[5])
    grad[6] = p[NS.index(x[0])]
    return grad

def func_log_sminus_t(x, p): # x = (n, d), p = (U0, U1, U2, U3, U4, s, t)
    return p[NS.index(x[0])] * (np.log(x[1] - np.exp(p[5])) + p[6])
def jac_log_sminus_t(x, p):
    grad = [0] * 7
    grad[NS.index(x[0])] = np.log(x[1] - np.exp(p[5])) + p[6]
    grad[5] = p[NS.index(x[0])] * 1 / (x[1] - np.exp(p[5])) * np.exp(p[5]) * -1
    grad[6] = p[NS.index(x[0])]
    return grad

def u_func_pow_r(x, p): # x = n, p = (W, r)
    return p[0] * x ** p[1]
def u_jac_pow_r(x, p):
    grad = [0] * 2
    grad[0] = x ** p[1]
    grad[1] = p[0] * x ** p[1] * np.log(x)
    return grad

def u_func_pow_r_t(x, p): # x = n, p = (W, r, t)
    return p[0] * (x ** p[1] + p[2])
def u_jac_pow_r_t(x, p):
    grad = [0] * 3
    grad[0] = x ** p[1] + p[2]
    grad[1] = p[0] * x ** p[1] * np.log(x)
    grad[2] = p[0]
    return grad

def u_func_log(x, p): # x = n, p = (W)
    return p[0] / np.log(x)
def u_jac_log(x, p):
    grad = [0] * 1
    grad[0] = 1 / np.log(x)
    return grad

def u_func_log_t(x, p): # x = n, p = (W, t)
    return p[0] / (np.log(x) + p[1])
def u_jac_log_t(x, p):
    grad = [0] * 2
    grad[0] = 1 / (np.log(x) + p[1])
    grad[1] = p[0] * -1 / (np.log(x) + p[1]) ** 2
    return grad


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

    sns.set_style("whitegrid")

    num_axs = 4
    fig, axs = plt.subplots(1, num_axs, figsize=(num_axs * 4, 3))

    rangee_by_ndc = {}

    for name in data_by_name:
        config = configs[name]
        data = data_by_name[name]
        const_data = const_data_by_name[name]
        ds = [d for d in data['ds'] if d in const_data['ds']]
        ys = [data['ys'][i] for i, d in enumerate(data['ds']) if d in ds]
        const_ys = [const_data['ys'][i] for i, d in enumerate(const_data['ds']) if d in ds]
        residues = np.array(ys) - np.array(const_ys)

        ax = axs[get_ax(name)]
        ax.scatter(
            ds,
            residues,
            color='white',
            edgecolors=config.color,
            label=config.label,
            s=5.0,
        )

        WARMUP_D_BY_N = {
            151898880: 150208512,
            319980544: 300154880,
            530074944: 530317312,
            681297408: 750256128,
            1176832000: 1000603648,
        }

        # overlay a cosine curve
        dmin = WARMUP_D_BY_N[data['ns'][0]]
        dmax = max(ds)
        half_period = dmax - dmin
        rangee = -np.mean(residues[-5:])
        cosine_ds = np.linspace(dmin, dmax, 100)
        cosine_ys = 0.5 * rangee * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1)
        # cosine_ys = 4.25e-4 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * cosine_ds ** 0.2 # good xC extrapolation
        # cosine_ys = 3.1e-5 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * cosine_ds ** 0.3
        # cosine_ys = 2.2e-6 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * cosine_ds ** 0.4
        # cosine_ys = 1.75e-7 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * cosine_ds ** 0.5
        # cosine_ys = 1.3e-8 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * cosine_ds ** 0.6 # good shape
        # cosine_ys = 1e-9 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * cosine_ds ** 0.7
        # cosine_ys = 7e-11 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * cosine_ds ** 0.8
        # cosine_ys = 4.6e-3 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * (np.log(cosine_ds) - 10)
        # cosine_ys = 11e-3 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * (np.log(cosine_ds) - 19)
        # cosine_ys = 12.5e-3 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * (np.log(cosine_ds) - 20) # good xC extrapolation
        # cosine_ys = 15e-3 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * (np.log(cosine_ds) - 21)
        # cosine_ys = 19e-3 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * (np.log(cosine_ds) - 22)
        # cosine_ys = 25e-3 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * (np.log(cosine_ds) - 23)
        # cosine_ys = 30e-3 * (np.cos(np.pi * (cosine_ds - dmin) / half_period) - 1) * (np.log(cosine_ds) - 23.5)
        ax.plot(
            cosine_ds,
            cosine_ys,
            color=config.color,
            linestyle="--",
            linewidth=1.0,
        )

        # # overlay an s2 curve
        # ds = data['ds']
        # multiplier = np.mean(residues[-5:]) / data['s2s'][-1]
        # s2s = [multiplier * s2 for s2 in data['s2s']]
        # ax.plot(
        #     ds,
        #     s2s,
        #     color=config.color,
        #     linestyle="-",
        #     linewidth=0.8,
        # )

        c = int(name.split('-')[-1][:-2])
        rangee_by_ndc[(data['ns'][0], ds[-1], c, name)] = rangee

    for ax in axs:
        ax.set_ylim(-0.20, 0.02)
        ax.legend(loc="upper right", ncols=1, fontsize=8)
        ax.set_xlabel("Tokens (D)")
    axs[0].set_ylabel(f"Residue")
    plt.suptitle("Residue of loss against curve of const LR schedule")
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")


    # plot the rangee
    if args.vfunc == '':
        exit()
    if args.vfunc == 'pow-r':
        func = func_pow_r
        jac = jac_pow_r
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
    elif args.vfunc == 'pow-r-t':
        func = func_pow_r_t
        jac = jac_pow_r_t
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
    elif args.vfunc == 'pow-r-splus-t':
        func = func_pow_r_splus_t
        jac = jac_pow_r_splus_t
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
    elif args.vfunc == 'pow-r-sminus-t':
        func = func_pow_r_sminus_t
        jac = jac_pow_r_sminus_t
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, 20.0), (None, None)]
    elif args.vfunc == 'log':
        func = func_log
        jac = jac_log
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None)]
    elif args.vfunc == 'log-t':
        func = func_log_t
        jac = jac_log_t
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
    elif args.vfunc == 'log-splus-t':
        func = func_log_splus_t
        jac = jac_log_splus_t
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
    elif args.vfunc == 'log-sminus-t':
        func = func_log_sminus_t
        jac = jac_log_sminus_t
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 0.0]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, 20.0), (None, None)]
    nds = [(n, d) for (n, d, c, name) in rangee_by_ndc.keys()]
    vs = list(rangee_by_ndc.values())
    coefficients, loss = get_coefficients_huber_nolog(nds, vs, func, jac, p0=p0, bounds=bounds)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    for (n, d, c, name), rangee in rangee_by_ndc.items():
        ax.scatter(d, rangee, color=configs[name].color, marker=MARKER_BY_C[c], s=80, label=configs[name].label)
    for n in NS:
        ds = [d for (nn, d, c, name) in rangee_by_ndc.keys() if nn == n]
        dmin, dmax = min(ds), max(ds)
        pred_ds = np.linspace(dmin, dmax, 100)
        pred_vs = [func((n, d), coefficients) for d in pred_ds]
        color = None
        for name, config in configs.items():
            if config.n == n:
                color = config.color
                break
        ax.plot(pred_ds, pred_vs, linestyle='--', color=color)
        ax.annotate(f'U={coefficients[NS.index(n)]:.4f}', xy=(dmax, pred_vs[-1]), xycoords='data', textcoords='offset points', xytext=(5, 0), va="center", ha="left", fontsize=10, color=color)
    if args.vfunc == 'pow-r':
        anno = f'v(D) = D^r\nr={coefficients[5]:.4f}'
    elif args.vfunc == 'pow-r-t':
        anno = f'v(D) = D^r + t\nr={coefficients[5]:.4f}, t={coefficients[6]:.4f}'
    elif args.vfunc == 'pow-r-splus-t':
        anno = f'v(D) = (D + exp(s))^r + t\nr={coefficients[5]:.4f}, s={coefficients[6]:.4f}, t={coefficients[7]:.4f}'
    elif args.vfunc == 'pow-r-sminus-t':
        anno = f'v(D) = (D - exp(s))^r + t\nr={coefficients[5]:.4f}, s={coefficients[6]:.4f}, t={coefficients[7]:.4f}'
    elif args.vfunc == 'log':
        anno = f'v(D) = log(D)'
    elif args.vfunc == 'log-t':
        anno = f'v(D) = log(D) + t\nt={coefficients[5]:.4f}'
    elif args.vfunc == 'log-splus-t':
        anno = f'v(D) = log(D + exp(s)) + t\ns={coefficients[5]:.4f}, t={coefficients[6]:.4f}'
    elif args.vfunc == 'log-sminus-t':
        anno = f'v(D) = log(D - exp(s)) + t\ns={coefficients[5]:.4f}, t={coefficients[6]:.4f}'
    anno += f'\nHuber loss = {loss:.4e}'
    ax.annotate(anno, xy=(0.3, 0.2), xycoords='axes fraction', va="top", ha="left", fontsize=10)
    ax.set_ylim(0.08, 0.23)
    ax.set_xlabel("Tokens (D)")
    ax.set_ylabel("Range of residue")
    ax.set_title(f"Range of residue, V = U * v(D)")
    ax.legend(ncol=5, fontsize=6, markerscale=0.5, loc='upper right')
    plt.savefig(args.output_path.replace('residue', f'residue_rangee_{args.vfunc}'), dpi=300, bbox_inches="tight")

    if args.ufunc == '':
        exit()
    if args.ufunc == 'pow-r':
        func = u_func_pow_r
        jac = u_jac_pow_r
        p0 = [0.0, 0.0]
        bounds = [(None, None), (None, None)]
    elif args.ufunc == 'pow-r-t':
        func = u_func_pow_r_t
        jac = u_jac_pow_r_t
        p0 = [0.0, 0.0, 0.0]
        bounds = [(None, None), (None, None), (None, None)]
    elif args.ufunc == 'log':
        func = u_func_log
        jac = u_jac_log
        p0 = [0.0]
        bounds = [(None, None)]
    elif args.ufunc == 'log-t':
        func = u_func_log_t
        jac = u_jac_log_t
        p0 = [0.0, 0.0]
        bounds = [(None, None), (None, None)]
    ns = NS
    us = coefficients[:len(ns)]
    coefficients, loss = get_coefficients_huber_nolog(ns, us, func, jac, p0=p0, bounds=bounds)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.scatter(ns, us, marker='o', s=80, color='black')
    nmin, nmax = min(ns), max(ns)
    pred_ns = np.linspace(nmin, nmax, 100)
    pred_us = [func(n, coefficients) for n in pred_ns]
    ax.plot(pred_ns, pred_us, linestyle='--', color='black')
    if args.ufunc == 'pow-r':
        anno = f'u(N) = n^r\nW={coefficients[0]:.4f}, r={coefficients[1]:.4f}'
    elif args.ufunc == 'pow-r-t':
        anno = f'u(N) = n^r + t\nW={coefficients[0]:.4f}, r={coefficients[1]:.4f}, t={coefficients[2]:.4f}'
    elif args.ufunc == 'log':
        anno = f'u(N) = 1 / log(n)\nW={coefficients[0]:.4f}'
    elif args.ufunc == 'log-t':
        anno = f'u(N) = 1 / (log(n) + t)\nW={coefficients[0]:.4f}, t={coefficients[1]:.4f}'
    anno += f'\nHuber loss = {loss:.4e}'
    ax.annotate(anno, xy=(0.4, 0.6), xycoords='axes fraction', va="top", ha="left", fontsize=10)
    ax.set_xlabel("Model size (N)")
    ax.set_ylabel("U")
    ax.set_title(f"U = W * u(N)")
    plt.savefig(args.output_path.replace('residue', f'residue_rangee_{args.vfunc}_{args.ufunc}'), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
