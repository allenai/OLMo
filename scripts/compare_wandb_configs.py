import logging
from collections.abc import MutableMapping

import click

from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def flatten(dictionary, parent_key="", separator="."):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


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
    left_run = api.run(left_run_path)
    right_run = api.run(right_run_path)

    left_config = flatten(left_run._attrs["rawconfig"])
    right_config = flatten(right_run._attrs["rawconfig"])

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
