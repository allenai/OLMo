import concurrent
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import click
from google.cloud import storage
from tqdm import tqdm

from olmo.util import prepare_cli_environment

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
    files_or_directories: Tuple[Path, ...],
):
    """
    Uploads artifacts to GCS. This uploads to a hardcoded bucket in GCS, because that's where we expect to keep all the artifacts for OLMo.

    WANDB_RUN_PATH: The "Weights and Biases" run path. You get this by going to the run in wandb and clicking on the "copy run path" button. We will use this as the prefix for the paths in the GCS bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket("ai2-olmo", user_project="ai2-olmo")
    prefix = wandb_run_path.strip("/")

    # Resolve directories to their contents; We only upload files.
    artifacts_to_upload = [
        (file_or_directory, prefix + "/" + file_or_directory.resolve().name)
        for file_or_directory in files_or_directories
    ]
    files_to_upload = []
    while len(artifacts_to_upload) > 0:
        file_or_directory, key = artifacts_to_upload.pop()
        if file_or_directory.is_file():
            files_to_upload.append((file_or_directory, key))
        elif file_or_directory.is_dir():
            for directory_entry in file_or_directory.iterdir():
                artifacts_to_upload.append((directory_entry, key + "/" + directory_entry.name))
        del key

    # Upload files in parallel
    def upload(file, key):
        blob = bucket.blob(key)
        with file.open("rb") as f:
            with tqdm.wrapattr(
                f,
                "read",
                total=file.stat().st_size,
                miniters=1,
                desc=f"Uploading {file} to gs://{bucket.name}/{key}",
            ) as f:
                blob.upload_from_file(f)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(upload, *args) for args in files_to_upload]
        for future in concurrent.futures.as_completed(futures):
            future.result()     # We do this so we can see exceptions.


if __name__ == "__main__":
    prepare_cli_environment()
    main()
