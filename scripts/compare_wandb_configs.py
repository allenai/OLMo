"""

Examples:
    Comparing Peteish7 to OLMoE
    - python scripts/compare_wandb_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39 https://wandb.ai/ai2-llm/olmoe/runs/rzsn9tlc

    Comparing Peteish7 to Amberish7
    - python scripts/compare_wandb_configs.py https://wandb.ai/ai2-llm/olmo-medium/runs/cej4ya39 https://wandb.ai/ai2-llm/olmo-medium/runs/ij4ls6v2
    

"""

import logging
import os
import re
from collections import Counter

import click

from olmo.util import flatten_dict, prepare_cli_environment

log = logging.getLogger(__name__)
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


def print_keys_with_differences(left_config, right_config):
    s_left = ""
    left_only_keys = left_config.keys() - right_config.keys()
    if len(left_only_keys) > 0:
        s_left += "Settings only in left:\n"
        s_left += "\n".join(f"\t{k}: {left_config[k]}" for k in sorted(left_only_keys)) + "\n"

    s_right = ""
    right_only_keys = right_config.keys() - left_config.keys()
    if len(right_only_keys) > 0:
        s_right += "Settings only in right:\n"
        s_right += "\n".join(f"\t{k}: {right_config[k]}" for k in sorted(right_only_keys)) + "\n"

    s_shared = ""
    keys_with_differences = {
        k for k in left_config.keys() & right_config.keys() if left_config[k] != right_config[k]
    }
    if len(keys_with_differences) > 0:
        for k in sorted(keys_with_differences):
            s_shared += f"{k}\n\t{left_config[k]}\n" + f"\t{right_config[k]}\n\n"

    if (s_left or s_right) and not s_shared:
        s = s_left + "=" * 50 + "\n" + s_right + "=" * 50 + "\n" + "No differences in shared settings.\n"
    else:
        s = s_left + "=" * 50 + "\n" + s_right + "=" * 50 + "\n" + s_shared
    print(s.strip())


def print_data_differences(left_data_paths: Counter, right_data_paths: Counter):
    print("===== Data Paths for left config:\n")
    simplified_left_data_paths = {path: count for path, count in left_data_paths.items()}
    for path, num_files in simplified_left_data_paths.items():
        print(f"\t{path}: {num_files}")
    print("\n\n")

    print("===== Data Paths for right config:\n")
    simplified_right_data_paths = {path: count for path, count in right_data_paths.items()}
    for path, num_files in simplified_right_data_paths.items():
        print(f"\t{path}: {num_files}")


@click.command()
@click.argument(
    "left_run_path",
    type=str,
)
@click.argument(
    "right_run_path",
    type=str,
)
def main(
    left_run_path: str,
    right_run_path: str,
):
    import wandb

    api = wandb.Api()
    left_run = api.run(parse_run_path(left_run_path))
    right_run = api.run(parse_run_path(right_run_path))

    left_config_raw = left_run._attrs["rawconfig"]
    right_config_raw = right_run._attrs["rawconfig"]

    # flattening the dict will make diffs easier
    left_config = flatten_dict(left_config_raw)
    right_config = flatten_dict(right_config_raw)

    # there are 2 specific fields in config that are difficult to diff:
    #   "evaluators" is List[Dict]
    #   "data.paths" is List[str]
    # let's handle each of these directly.

    # first, data.paths can be grouped and counted.
    left_data_paths = Counter([os.path.dirname(path) for path in left_config["data.paths"]])
    right_data_paths = Counter([os.path.dirname(path) for path in right_config["data.paths"]])
    del left_config["data.paths"]
    del right_config["data.paths"]

    # next, evaluators can be added to the flat dict with unique key per evaluator
    # also, each evaluator can also have a 'data.paths' field which needs collapsing
    def _simplify_evaluator(evaluator):
        evaluator = flatten_dict(evaluator)
        if evaluator["data.paths"]:
            evaluator["data.paths"] = Counter([os.path.dirname(path) for path in evaluator["data.paths"]])
        return evaluator

    def _simplify_evaluators(evaluators):
        simplified_evaluators = {}
        for evaluator in evaluators:
            new_key = (".".join(["evaluators" + "." + evaluator["type"] + "." + evaluator["label"]])).upper()
            simplified_evaluators[new_key] = _simplify_evaluator(evaluator)
        return simplified_evaluators

    left_evaluators = flatten_dict(_simplify_evaluators(left_config["evaluators"]), separator="___")
    right_evaluators = flatten_dict(_simplify_evaluators(right_config["evaluators"]), separator="___")
    del left_config["evaluators"]
    del right_config["evaluators"]

    print(
        f"==================== Config differences between {left_run_path} and {right_run_path} ====================\n\n"
    )

    # print config differences
    print("==================== Param differences ====================\n\n")
    print_keys_with_differences(left_config=left_config, right_config=right_config)
    print("============================================================= \n\n")

    # print data differences
    print("==================== Data Differences ====================\n\n")
    print_data_differences(left_data_paths, right_data_paths)
    print("============================================================= \n\n")

    # print eval differences
    print("==================== Eval Differences ====================\n\n")
    print_keys_with_differences(left_config=left_evaluators, right_config=right_evaluators)
    print("============================================================= \n\n")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
