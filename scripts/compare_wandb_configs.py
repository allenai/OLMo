import logging
import re

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


def print_keys_with_differences(left_config, right_config, level=0):
    prefix = "\t\t" * level

    s_left = ""
    left_only_keys = left_config.keys() - right_config.keys()
    if len(left_only_keys) > 0:
        s_left += prefix + "Settings only in left:\n"
        s_left += (prefix + "\n").join(f"\t{k}: {left_config[k]}" for k in sorted(left_only_keys)) + "\n"

    s_right = ""
    right_only_keys = right_config.keys() - left_config.keys()
    if len(right_only_keys) > 0:
        s_right += prefix + "Settings only in right:\n"
        s_right += (prefix + "\n").join(f"\t{k}: {right_config[k]}" for k in sorted(right_only_keys)) + "\n"

    s_shared = ""
    keys_with_differences = {
        k for k in left_config.keys() & right_config.keys() if left_config[k] != right_config[k]
    }
    if len(keys_with_differences) > 0:
        for k in sorted(keys_with_differences):
            if isinstance(left_config[k], list) and isinstance(right_config[k], list):
                s_list = prefix + f"{k}:\n"
                for left, right in zip(left_config[k], right_config[k]):  # assumes lists are same order
                    if isinstance(left, dict) and isinstance(right, dict):
                        print_keys_with_differences(left_config=left, right_config=right, level=level + 1)
                    else:
                        s_list += prefix + f"\t{left}\n" + prefix + f"\t{right}\n\n"
                if s_list != prefix + f"{k}:\n":
                    s_shared += s_list
            else:
                s_shared += prefix + f"{k}\n\t{left_config[k]}\n" + prefix + f"\t{right_config[k]}\n\n"

    if (s_left or s_right) and not s_shared:
        s = s_left + s_right + prefix + "No differences in shared settings.\n"
    else:
        s = s_left + s_right + s_shared
    print(s.strip())
    return


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

    left_config = flatten_dict(left_run._attrs["rawconfig"], include_lists=True)
    right_config = flatten_dict(right_run._attrs["rawconfig"], include_lists=True)

    print_keys_with_differences(left_config=left_config, right_config=right_config)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
