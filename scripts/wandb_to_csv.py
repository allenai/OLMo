import logging
from pathlib import Path
from typing import Union, Dict, Any
import time

import click

log = logging.getLogger(__name__)


def dump_run(run, output_file: Path):
    # open output file and check what's already there
    data: Dict[int, Dict[str, Any]] = {}
    if output_file.is_file():
        f = output_file.open("r+", encoding="UTF-8")
        lines = iter(f)
        try:
            columns = next(lines).rstrip("\n").split("\t")
        except StopIteration:
            columns = None
        if columns is not None:
            step_column = columns.index("_step")
            for line in lines:
                line = line.rstrip("\n").split("\t")
                step = int(line[step_column])
                data[step] = {
                    column_name: line[i]
                    for i, column_name in enumerate(columns)
                }
    else:
        f = output_file.open("x", encoding="UTF-8")

    def write_out_results():
        f.seek(0)

        # find columns
        columns = set()
        for values in data.values():
            columns |= values.keys()
        columns = list(columns)
        columns.sort()

        # write data
        ordered_data = list(data.items())
        ordered_data.sort()

        f.truncate()
        f.write("\t".join(columns))
        f.write("\n")

        with click.progressbar(ordered_data) as bar:
            for step, values in bar:
                line_values = (values.get(column) for column in columns)
                line_values = (str(v) if v is not None else "" for v in line_values)
                line = "\t".join(line_values)
                line += "\n"
                f.write(line)

    # scan ranges of steps
    current_step = 0
    last_write = time.time()
    while True:
        while current_step in data.keys():
            current_step += 1
        with click.progressbar(run.scan_history(min_step=current_step)) as bar:
            for s in bar:
                step = s["_step"]
                if step in data:
                    break
                data[step] = s
                current_step = max(current_step, step)
            else:
                break

            # If we've been reading for 10 minutes, write out results
            if time.time() - last_write > 60 * 10:
                write_out_results()
                last_write = time.time()

    write_out_results()

    f.close()


@click.group()
def cli():
    pass


@cli.command(name="dump-run")
@click.argument("wandb_run_path", type=str)
@click.argument("output_file", type=str)
def dump_run_cli(wandb_run_path: str, output_file: Union[str, Path]):
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)
    output_file = Path(output_file)
    dump_run(run, output_file)


@cli.command(name="dump-group")
@click.argument("wandb_project", type=str)
@click.argument("wandb_group", type=str)
@click.argument("output_dir", type=str)
def dump_group_cli(wandb_project: str, wandb_group: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    api = wandb.Api()
    runs = api.runs(f"ai2-llm/{wandb_project}", filters={"group": wandb_group}, order="created_at")
    for run in runs:
        filename = f"{run.path[-1]}.csv"
        log.info("Writing %s", filename)
        dump_run(run, output_dir / filename)


@cli.command(name="merge-files")
@click.argument("input_files", nargs=-1, type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
def merge_files(input_files, output_file):
    data: Dict[int, Dict[str, Any]] = {}
    for input_file in input_files:
        lines = iter(input_file)
        try:
            columns = next(lines).rstrip("\n").split("\t")
        except StopIteration:
            continue
        step_column = columns.index("_step")
        for line in lines:
            line = line.rstrip("\n").split("\t")
            step = int(line[step_column])
            data[step] = {
                column_name: line[i]
                for i, column_name in enumerate(columns)
            }

    # find columns
    columns = set()
    for values in data.values():
        columns |= values.keys()
    columns = list(columns)
    columns.sort()

    # write data
    ordered_data = list(data.items())
    ordered_data.sort()

    output_file.write("\t".join(columns))
    output_file.write("\n")

    with click.progressbar(ordered_data) as bar:
        for step, values in bar:
            line_values = (values.get(column) for column in columns)
            line_values = (str(v) if v is not None else "" for v in line_values)
            line = "\t".join(line_values)
            line += "\n"
            output_file.write(line)

    output_file.close()


if __name__ == "__main__":
    from olmo.util import prepare_cli_environment
    prepare_cli_environment()
    cli()
