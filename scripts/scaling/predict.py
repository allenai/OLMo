# python scripts/scaling/predict.py -k main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -n 6887575552 -d 3945065873408 -t 7b
# python scripts/scaling/predict.py -k main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -n 13202396160 -d 5000080130048 -t 13b
# python scripts/scaling/predict.py -k main_mc -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2_mc.json -y mc_acc -n 6887575552 -d 3945065873408 -t 7b-4T-final
# python scripts/scaling/predict.py -k main_mc -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2_mc.json -y mc_acc  -n 13202396160 -d 5000080130048 -t 13b-5T-final

import argparse

import numpy as np
from step1 import fit_step1
from step2 import fit_step2

from olmo.scaling.scaling_laws.fitting_functions import chinchilla_n_d_fit, sigmoid
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_step2_data_by_name,
    get_task_sets,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--keys", nargs="+", default=[], help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument(
        "-y", "--y_metric", default="rc_acc", choices=["rc_acc", "mc_acc"], help="Metric to predict"
    )
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument(
        "--skip_perc",
        type=float,
        default=0.0,
        help="Percentage of intermediate ckpts to skip from the beginning (for loss to accuracy fitting)",
    )
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("--step2-config-path", type=str, default=None, help="Path to config file for step2")
    parser.add_argument("-n", "--n", type=int, required=True, help="Model size of the target model")
    parser.add_argument("-d", "--d", type=int, required=True, help="Data size of the target model")
    parser.add_argument(
        "-t", "--target-name", type=str, default=None, help="Path to the csv file of the target model"
    )
    args = parser.parse_args()

    args.keys = get_task_sets(args.keys)

    return args


def main():
    args = parse_args()
    configs = get_final_configs(args.config_path)
    if args.step2_config_path:
        step2_configs = get_final_configs(args.step2_config_path)
    else:
        step2_configs = configs

    results = "Task Name | Prediction | Actual | Rel Error"

    for r, task_name in enumerate(args.keys):
        # Step 1
        step1_data_by_name = get_step1_data_by_name(
            configs, task_name, y_metric="rc_bpb", moving_avg=args.moving_avg
        )
        step1_coefficients = fit_step1(step1_data_by_name, y_metric="rc_bpb")

        # Step 2
        step2_data_by_name = get_step2_data_by_name(
            step2_configs, task_name, y_metric=args.y_metric, moving_avg=args.moving_avg, skip_perc=args.skip_perc
        )
        step2_coefficients, _ = fit_step2(step2_data_by_name, task_name, args.y_metric)

        # make predictions
        pred_loss = chinchilla_n_d_fit([args.n, args.d], step1_coefficients)
        pred_acc = sigmoid(pred_loss, *step2_coefficients)
        if args.target_name:
            data = step2_data_by_name[args.target_name]
            actual_acc = data["ys"][-1]
            rel_error = np.abs(pred_acc - actual_acc) / actual_acc
            results += f"\n{task_name} | {pred_acc * 100:.1f} | {actual_acc * 100:.1f} | {rel_error * 100:.1f}%"
        else:
            results += f"\n{task_name} | {pred_acc * 100:.1f} | - | -"

    print(results)


if __name__ == "__main__":
    main()
