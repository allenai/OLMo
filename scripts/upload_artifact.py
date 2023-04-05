import logging
from pathlib import Path
from typing import Tuple

import click
from google.cloud import storage
from tqdm import tqdm

from dolma.util import prepare_cli_environment

log = logging.getLogger(__name__)


@click.command()
@click.argument(
    "wandb_run_path",
    type=str,
)
@click.argument(
    "files_or_directories",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
)
def main(
    wandb_run_path: str,
    files_or_directories: Tuple[Path],
):
    storage_client = storage.Client()
    bucket = storage_client.bucket("allennlp-olmo", "ai2-allennlp")
    prefix = wandb_run_path.strip("/")

    files_or_directories_in_a_special_variable_because_mypy_is_lame = [
        (file_or_directory, prefix + "/" + file_or_directory.name) for file_or_directory in files_or_directories
    ]
    while len(files_or_directories_in_a_special_variable_because_mypy_is_lame) > 0:
        file_or_directory, key = files_or_directories_in_a_special_variable_because_mypy_is_lame.pop()
        if file_or_directory.is_file():
            blob = bucket.blob(key)
            with file_or_directory.open("rb") as f:
                with tqdm.wrapattr(
                    f,
                    "read",
                    total=file_or_directory.stat().st_size,
                    miniters=1,
                    desc=f"Uploading {file_or_directory} to gs://{bucket.name}/{key}",
                ) as f:
                    blob.upload_from_file(f, file_or_directory)
        elif file_or_directory.is_dir():
            for directory_entry in file_or_directory.iterdir():
                files_or_directories_in_a_special_variable_because_mypy_is_lame.append(
                    (directory_entry, key + "/" + directory_entry.name)
                )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
