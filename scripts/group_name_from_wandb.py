import logging

import click

from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


@click.command()
@click.argument(
    "wandb_run_path",
    type=str,
)
def main(wandb_run_path: str):
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)
    print(run.group)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
