import os
import sys
from typing import Optional, Tuple

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from efficiency_benchmark.steps import (
    CalculateMetricsStep,
    LogOutputStep,
    PredictStep,
    TabulateMetricsStep,
)
from gantry import run as gantry_run

_CLICK_GROUP_DEFAULTS = {
    "cls": HelpColorsGroup,
    "help_options_color": "green",
    "help_headers_color": "yellow",
    "context_settings": {"max_content_width": 115},
}

_CLICK_COMMAND_DEFAULTS = {
    "cls": HelpColorsCommand,
    "help_options_color": "green",
    "help_headers_color": "yellow",
    "context_settings": {"max_content_width": 115},
}


@click.group(**_CLICK_GROUP_DEFAULTS)
def main():
    pass


@main.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument("cmd", nargs=-1)
@click.option(
    "-t",
    "--task",
    type=str,
    nargs=1,
    help="""Tasks.""",
)
@click.option(
    "--split",
    type=str,
    help="""Split.""",
    default="test",
)
@click.option(
    "-s",
    "--scenario",
    type=str,
    default="single_stream",
    help="""Evaluation scenario [single_stream, random_batch, offline].""",
)
@click.option(
    "-b",
    "--max_batch_size",
    type=int,
    default=32,
    help="""Maximum batch size.""",
)
@click.option(
    "-o",
    "--offline_dir",
    type=str,
    nargs=1,
    help="""Offline dir.""",
)
@click.option(
    "--output_dir",
    type=str,
    nargs=1,
    help="""Output folder.""",
)
@click.option(
    "-l",
    "--limit",
    type=int,
    default=-1,
    help="""Limit.""",
)
def run(
    cmd: Tuple[str, ...],
    task: str,
    split: str = "test",
    scenario: str = "accuracy",
    max_batch_size: int = 32,
    offline_dir: str = f"{os.getcwd()}/datasets/efficiency-beenchmark",
    limit: Optional[int] = -1,
    output_dir: Optional[str] = None,
):
    if scenario == "offline":
        try:
            os.makedirs(offline_dir, exist_ok=True)
        except:
            sys.exit(f"Failed to write to offline directory: {offline_dir}.")

    metric_task_dict = {}
    prediction_step = PredictStep(
        cmd=cmd,
        task=task,
        scenario=scenario,
        max_batch_size=max_batch_size,
        offline_dir=offline_dir,
        split=split,
        limit=limit,
    )
    if output_dir:
        output_dir = prediction_step.task.base_dir(base_dir=output_dir)
        try:
            os.makedirs(f"{output_dir}/{scenario}/", exist_ok=True)
            print(f"Output to: {output_dir}/{scenario}/")
        except OSError:
            print(f"Failed to create output directory: {output_dir}. Logging to STDOUT.")
            output_dir = None
    predictions, metrics = prediction_step.run()
    if scenario == "accuracy":
        metric_step = CalculateMetricsStep(task=task)
        acc_metrics = metric_step.calculate_metrics(predictions=predictions)
        metric_task_dict[task] = acc_metrics
        if len(acc_metrics.keys()) > 0:
            metrics["accuracy"] = acc_metrics
        output_step = LogOutputStep(
            task=task, output_file=f"{output_dir}/{scenario}/outputs.json" if output_dir else None
        )
        output_step.run(predictions=predictions)

    table_step = TabulateMetricsStep()
    table_step_result = table_step.run(metrics=metric_task_dict)

    print("\n".join(table_step_result))
    prediction_step.tabulate_efficiency_metrics(
        metrics, output_file=f"{output_dir}/{scenario}/metrics.json" if output_dir else None
    )


@main.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument("cmd", nargs=-1)
@click.option(
    "-t",
    "--task",
    type=str,
    nargs=1,
    help="""Tasks.""",
)
@click.option(
    "--split",
    type=str,
    help="""Split.""",
)
@click.option(
    "-l",
    "--limit",
    type=int,
    default=None,
    help="""Limit.""",
)
@click.option(
    "-b",
    "--max_batch_size",
    type=int,
    default=32,
    help="""Maximum batch size.""",
)
@click.option(
    "--cpus",
    type=float,
    help="""Minimum number of logical CPU cores (e.g. 4.0, 0.5).""",
)
@click.option(
    "--dataset",
    type=str,
    multiple=True,
    help="""An input dataset in the form of 'dataset-name:/mount/location' to attach to your experiment.
    You can specify this option more than once to attach multiple datasets.""",
)
def submit(
    cmd: Tuple[str, ...],
    task: str,
    split: str = "validation",
    limit: int = None,
    max_batch_size: int = 32,
    cpus: Optional[float] = None,
    dataset: Optional[Tuple[str, ...]] = None,
):
    gantry_run(
        arg=cmd,
        task=task,
        split=split,
        limit=limit,
        max_batch_size=max_batch_size,
        cluster=["efficiency-benchmark/elanding-rtx-8000"],  # TODO
        beaker_image="haop/efficiency-benchmark",  # TODO
        workspace="efficiency-benchmark/efficiency-benchmark",
        cpus=cpus,
        gpus=2,  # hard code to 2 to make sure only one job runs at a time.
        allow_dirty=True,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
