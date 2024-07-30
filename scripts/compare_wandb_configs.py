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

    left_config = flatten_dict(left_run._attrs["rawconfig"])
    right_config = flatten_dict(right_run._attrs["rawconfig"])

    left_only_keys = left_config.keys() - right_config.keys()
    if len(left_only_keys) > 0:
        print("Settings only in left:")
        print("\n".join(f"\t{k}: {left_config[k]}" for k in sorted(left_only_keys)))
        print()

    right_only_keys = right_config.keys() - left_config.keys()
    if len(right_only_keys) > 0:
        print("Settings only in right:")
        print("\n".join(f"\t{k}: {right_config[k]}" for k in sorted(right_only_keys)))
        print()

    keys_with_differences = {
        k for k in left_config.keys() & right_config.keys() if left_config[k] != right_config[k]
    }
    if len(keys_with_differences) > 0:
        if len(left_only_keys) > 0 or len(right_only_keys) > 0:
            print("Settings with differences:")
        print("\n".join(f"{k}\n\t{left_config[k]}\n\t{right_config[k]}\n" for k in sorted(keys_with_differences)))
    else:
        print("No differences in shared settings.")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
