import logging
from pathlib import Path
from typing import Tuple

from google.cloud import storage

import click

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

    files_or_directories = [
        (file_or_directory, prefix + "/" + file_or_directory.name)
        for file_or_directory in files_or_directories
    ]
    while len(files_or_directories) > 0:
        file_or_directory, key = files_or_directories.pop()
        if file_or_directory.is_file():
            blob = bucket.blob(key)
            log.info(f"Uploading {file_or_directory} to gs://{bucket.name}/{key}")
            blob.upload_from_filename(file_or_directory)
        elif file_or_directory.is_dir():
            for f in file_or_directory.iterdir():
                files_or_directories.append((f, key + "/" + f.name))


if __name__ == "__main__":
    prepare_cli_environment()
    main()
