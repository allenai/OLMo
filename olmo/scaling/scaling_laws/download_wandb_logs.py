import argparse
import csv
import json
import re
from typing import List

import wandb
from tqdm import tqdm

from olmo.scaling.scaling_laws.utils import downstream, v3_validation, validation

run_path_re = re.compile(r"^[^/]+/[^/]+/[^/]+$")
run_path_url = re.compile(r"^https?://wandb.ai/([^/]+)/([^/]+)/runs/([^/]+)")


def parse_run_path(run_path: str) -> str:
    """For convenience, we allow run paths as well as URLs."""
    run_path = run_path.strip("/")
    if run_path_re.match(run_path):
        return run_path

    m = run_path_url.match(run_path)
    if m is not None:
        entity, project, run_id = m.groups()
        return f"{entity}/{project}/{run_id}"

    raise ValueError(f"Could not parse '{run_path}'")


def get_runs(run_paths: List) -> List:
    all_wb_runs: List = []
    api = wandb.Api()
    for run_path in run_paths:
        run_path = parse_run_path(run_path)
        run_name = run_path.split("/")[-1]
        wb_path = run_path.replace("/" + run_name, "")
        wb_filters = {"$or": [{"display_name": (n if "*" not in n else {"$regex": n})} for n in [run_name]]}
        wb_runs = api.runs(path=wb_path, filters=wb_filters)
        print(f"Found {len(wb_runs)} matching runs in {wb_path}")
        all_wb_runs += wb_runs
    return all_wb_runs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--wandb-names", type=str, nargs="+", required=True, help="Full run name or regex")
    parser.add_argument("-x", "--x-axis", type=str, default="throughput/total_tokens", help="X axis")
    parser.add_argument("-y", "--y-axis", nargs="+", type=str, default=["train/Perplexity"], help="Y axis")
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output csv file",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    if args.y_axis == ["eval/all-validation/CrossEntropyLoss"]:
        args.y_axis = [f"eval/{d}/CrossEntropyLoss" for d in validation]

    elif args.y_axis == ["eval/all-v3-validation/CrossEntropyLoss"]:
        args.y_axis = [f"eval/{d}/CrossEntropyLoss" for d in v3_validation]

    elif args.y_axis == ["eval/downstream/all"]:
        args.y_axis = [f"eval/downstream/{d}" for d in downstream]

    wb_runs = get_runs(args.wandb_names)

    print("Downloading the data from the following wandb runs:\n", "\n".join([str(run) for run in wb_runs]))
    print(wb_runs)

    with open(args.output_path, "w") as file_ref:
        writer = csv.DictWriter(file_ref, fieldnames=[args.x_axis] + args.y_axis + ["batch_size_in_tokens"])
        writer.writeheader()

        rows = []
        for wb_run in tqdm(wb_runs):
            print(f"Processing {wb_run.name}")
            history = wb_run.scan_history(
                keys=[args.x_axis] + args.y_axis, page_size=10000
            )  # page_size cannot be too big, it will make it faster but it will start to downsample

            config = json.loads(wb_run.json_config)
            batch_size_in_tokens = (
                config["global_train_batch_size"]["value"] * config["model"]["value"]["max_sequence_length"]
            )

            for wb_step in history:
                rows.append(wb_step)
                # With certain run restarts, we also update the batch size.
                wb_step["batch_size_in_tokens"] = batch_size_in_tokens

        rows = sorted(rows, key=lambda x: x[args.x_axis])
        writer.writerows(rows)


if __name__ == "__main__":
    # python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-mup/new_mup_olmo_256*' -y train/CrossEntropyLoss -o wandb_outputs/mup-olmo-256-train.csv
    main()
