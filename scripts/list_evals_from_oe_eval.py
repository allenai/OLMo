import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from olmo.eval.downstream import METRIC_FROM_OE_EVAL
from olmo.util import load_oe_eval_requests
from olmo_data.data import get_data_path


def show_oe_eval_task(config_file, dir, num_examples, sub_dir=None):
    if not os.path.isfile(config_file):
        return []
    if num_examples > 0:
        config, requests = load_oe_eval_requests(dir, sub_dir)
    else:
        config = read_json(config_file)
    default_metric = METRIC_FROM_OE_EVAL.get(config["task_config"]["primary_metric"])
    task_name = dir
    if sub_dir:
        task_name += f"_{sub_dir}"
    print(
        f"Task name: {task_name}  Default metric: {default_metric}  Num instances: {config['num_instances']} "
        + f"Full task config: {config['task_config']}"
    )
    if num_examples > 0:
        print("Example requests:")
        for i, example in enumerate(requests):
            if i >= num_examples:
                break
            print(f"Example {i}: {example}")
    print("-------------------------------")
    ds_name_str = ""
    if sub_dir:
        ds_name_str = f'"dataset_name": "{sub_dir}", '
    res = [
        f'    "{task_name}": (OEEvalTask, {{"dataset_path": "{dir}", {ds_name_str}"metric_type": "{default_metric}"}}),',
        f'    "{task_name}_bpb": (OEEvalTask, {{"dataset_path": "{dir}", {ds_name_str}"metric_type": "bpb"}}),',
    ]
    return res


def list_oe_eval_tasks(num_examples=0):
    with get_data_path("oe_eval_tasks") as raw_dir:
        oe_eval_dir = str(raw_dir)
    if not os.path.exists(oe_eval_dir):
        raise FileNotFoundError(f"Could not find OE eval directory at {oe_eval_dir}")
    all_configs = []
    for dir in sorted(os.listdir(oe_eval_dir)):
        for sub_dir in sorted(os.listdir(os.path.join(oe_eval_dir, dir))):
            config_file = os.path.join(oe_eval_dir, dir, sub_dir, "config.json")
            all_configs += show_oe_eval_task(config_file, dir, num_examples, sub_dir=sub_dir)
        config_file = os.path.join(oe_eval_dir, dir, "config.json")
        all_configs += show_oe_eval_task(config_file, dir, num_examples, sub_dir=None)
    print("\n==============\nInsertions to label_to_task_map:")
    print("\n".join(all_configs))
    print("\n==============\nInsertions in training configs:")
    for config in all_configs:
        task_name = re.findall(r'^    "([^"]*)"', config)[0]
        print(f"  - label: {task_name}\n    type: downstream\n\n")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-examples",
        type=int,
        default=0,
        help="Number of example requests to show for each task.",
    )
    args = parser.parse_args()
    list_oe_eval_tasks(
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()
