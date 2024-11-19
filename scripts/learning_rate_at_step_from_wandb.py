import logging

import click

from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


@click.command()
@click.argument(
    "wandb_run_path",
    type=str,
)
@click.argument(
    "step",
    type=int,
)
def main(wandb_run_path: str, step: int):
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)
    learning_rate_key = "optim/learning_rate_group0"
    for data in run.scan_history(keys=["_step", learning_rate_key], min_step=step - 1, max_step=step + 1):
        data_step = int(data["_step"])
        if data_step == step:
            lr = data[learning_rate_key]
            print(lr)
            break
    else:
        raise RuntimeError(f"Could not find step {step} in {wandb_run_path}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
