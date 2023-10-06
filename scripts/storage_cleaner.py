import argparse
import logging
import os
import re
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from argparse import ArgumentParser, _SubParsersAction
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import google.cloud.storage as gcs
import wandb
from google.api_core.exceptions import NotFound

from olmo import TrainConfig

log = logging.getLogger(__name__)


CONFIG_YAML: str = "config.yaml"
WANDB_ENTITY: str = "ai2-llm"
DEFAULT_MAX_ARCHIVE_SIZE: float = 5_000_000_000  # 5GB


class CleaningOperations(Enum):
    DELETE_BAD_RUNS = auto()
    RENAME_RUNS_TO_WANDB_ID = auto()


class StorageType(Enum):
    LOCAL_FS = auto()
    GCS = auto()
    S3 = auto()
    R2 = auto()


class StorageAdapter(ABC):
    @abstractmethod
    def list_entries(self, path: str, max_file_size: Optional[float] = None) -> List[str]:
        """List all the entries within the directory or compressed file at the given path.
        Returns only top-level entries (i.e. not entries in subdirectories).

        max_file_size sets a threshold for the largest size file to retain within entries.
        Any file of larger size is not included in the returned results.
        """

    @abstractmethod
    def list_dirs(self, path: str) -> List[str]:
        """List all the directories within the directory or compressed file at the given path.
        Returns only top-level entries (i.e. not entries in subdirectories).
        """

    @abstractmethod
    def delete_path(self, path: str):
        pass

    @abstractmethod
    def is_file(self, path: str):
        pass


class LocalFileSystemAdapter(StorageAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._temp_files: List[Any] = []
        self._archive_extensions: List[str] = []

    def __del__(self):
        for temp_file in self._temp_files:
            temp_file.close()

    def create_temp_file(self, suffix: str = "") -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
        self._temp_files.append(temp_file)
        return temp_file.name

    def has_supported_archive_extension(self, path: str) -> bool:
        if len(self._archive_extensions) == 0:
            self._archive_extensions = [
                extension.lower() for _, extensions, _ in shutil.get_unpack_formats() for extension in extensions
            ]

        return any(path.lower().endswith(extension) for extension in self._archive_extensions)

    def _list_entries(self, path: str, no_files: bool = False, max_file_size: Optional[float] = None) -> List[str]:
        if os.path.isdir(path):
            return [
                entry
                for entry in os.listdir(path)
                if ((max_file_size is None or os.path.getsize(entry) <= max_file_size)
                    and (not no_files or not os.path.isfile(os.path.join(path, entry))))
            ]

        if self.has_supported_archive_extension(path):
            if no_files or max_file_size is not None:
                raise NotImplementedError("Filtering out entries from a tar file is not yet supported")

            with tarfile.open(path) as tar:
                tar_subpaths = [os.path.normpath(name) for name in tar.getnames()]
                return [
                    os.path.basename(tar_subpath) for tar_subpath in tar_subpaths if tar_subpath.count(os.sep) == 1
                ]

        raise ValueError(f"Path does not correspond to directory or supported archive file: {path}")

    def list_entries(self, path: str, max_file_size: Optional[float] = None) -> List[str]:
        return self._list_entries(path, max_file_size=max_file_size)

    def list_dirs(self, path: str) -> List[str]:
        return self._list_entries(path, no_files=True)

    def delete_path(self, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            return

        if path_obj.is_file():
            path_obj.unlink()
        else:
            shutil.rmtree(path)

    def is_file(self, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            return

        return path_obj.is_file()


class GoogleCloudStorageAdapter(StorageAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._local_fs_adapter: Optional[LocalFileSystemAdapter] = None
        self._gcs_client: Optional[gcs.Client] = None
        self._temp_dirs: List[tempfile.TemporaryDirectory] = []

    @property
    def local_fs_adapter(self):
        if self._local_fs_adapter is None:
            self._local_fs_adapter = LocalFileSystemAdapter()

        return self._local_fs_adapter

    @property
    def gcs_client(self):
        if self._gcs_client is None:
            self._gcs_client = gcs.Client()

        return self._gcs_client
    
    @staticmethod
    def _get_bucket_name_and_key(path: str) -> Tuple[str, str]:
        parsed_path = urlparse(path)
        bucket_name = parsed_path.netloc
        key = parsed_path.path.lstrip("/")
        return bucket_name, key

    def _get_blob_size(self, blob: gcs.Blob) -> int:
        blob.reload()
        if blob.size is None:
            raise ValueError(f"Failed to get size for blob: {blob.name}")
        return blob.size

    def _is_file(self, bucket_name: str, key: str) -> bool:
        # print(bucket_name, key)
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(key)
        try:
            blob.reload()
            print(blob.name)
            return True
        except NotFound:
            return False

    def _get_size(self, bucket_name: str, key: str) -> int:
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.get_blob(key)
        if blob is None:
            raise ValueError(f"Getting size for invalid object with bucket | key: {bucket_name} | {key}")

        return self._get_blob_size(blob)

    def _download_file(self, bucket_name: str, key: str) -> str:
        extension = "".join(Path(key).suffixes)
        temp_file = self.local_fs_adapter.create_temp_file(suffix=extension)

        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.get_blob(key)
        if blob is None:
            raise ValueError(f"Downloading invalid object with bucket | key: {bucket_name} | {key}")
        blob.download_to_filename(temp_file)
        return temp_file

    def _get_directory_entries(self, bucket_name: str, key: str, no_files: bool = False, max_file_size: Optional[float] = None) -> List[str]:
        bucket = self.gcs_client.bucket(bucket_name)
        # Setting max_results to 10,000 as a reasonable caution that a directory should not have
        # more than 10,000 entries.
        # Using delimiter causes result to have directory-like structure
        blobs = bucket.list_blobs(max_results=10_000, prefix=key, delimiter="/")

        entries: List[str] = []
        for blob in blobs:
            blob: gcs.Blob

            if no_files:
                # Note: We need to iterate through (or otherwise act on?) the blobs to populate blob.prefixes
                # Thus we no-op here rather than skipping the loop
                continue

            size: int = self._get_blob_size(blob)
            if max_file_size is not None and size > max_file_size:
                log.info(
                    "Blob %s has size %.2fGb exceeding max file size %.2fGb, skipping.",
                    blob.name,
                    size / 1e9,
                    max_file_size / 1e9,
                )
                continue

            entries.append(blob.name)  # type: ignore

        # Note: We need to iterate through (or otherwise act on?) the blobs to populate blob.prefixes
        entries += blobs.prefixes

        return [entry.removeprefix(key) for entry in entries]

    def _list_entries(self, path: str, no_files: bool = False, max_file_size: Optional[float] = None) -> List[str]:
        bucket_name, key = self._get_bucket_name_and_key(path)

        if self.local_fs_adapter.has_supported_archive_extension(path):
            file_path = self._download_file(bucket_name, key)

            if no_files:
                return self.local_fs_adapter.list_dirs(file_path)
            return self.local_fs_adapter.list_entries(file_path, max_file_size)

        if self._is_file(bucket_name, key):
            # print(bucket_name, key)
            raise ValueError(f"Path corresponds to a file without a supported archive extension {path}")

        res = self._get_directory_entries(bucket_name, key, no_files=no_files, max_file_size=max_file_size)
        # print('Result', res)
        return res

    def list_entries(self, path: str, max_file_size: Optional[float] = None) -> List[str]:
        return self._list_entries(path, max_file_size=max_file_size)

    def list_dirs(self, path: str) -> List[str]:
        return self._list_entries(path, no_files=True)

    def delete_path(self, path: str):
        bucket_name, key = self._get_bucket_name_and_key(path)

        bucket = self.gcs_client.bucket(bucket_name)
        # Not using delimiter causes result to not have directory-like structure (all blobs returned)
        blobs = list(bucket.list_blobs(prefix=key))

        # blob_names = []
        # for blob in blobs:
        #     blob_names.append(blob.name)
        bucket.delete_blobs(blobs)

        # print(len(blob_names))
        raise NotImplementedError()
    
    def is_file(self, path: str):
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._is_file(bucket_name, key)


class StorageCleaner:
    def __init__(
        self,
        dry_run: bool = False,
        ignore_prompts: bool = False,
        runs_require_config_yaml: bool = True,
        max_archive_size: Optional[float] = None,
        default_wandb_entity: Optional[str] = None,
        default_wandb_project: Optional[str] = None,
    ) -> None:
        self._dry_run: bool = dry_run
        self._storage_adapters: Dict[StorageType, StorageAdapter] = {}
        self._runs_require_config_yaml = runs_require_config_yaml
        self._ignore_prompts: bool = ignore_prompts
        self._max_archive_size: Optional[float] = max_archive_size
        self._default_wandb_entity: Optional[str] = default_wandb_entity
        self._default_wandb_project: Optional[str] = default_wandb_project

    @staticmethod
    def _is_url(path: str) -> bool:
        return re.match(r"[a-z0-9]+://.*", str(path)) is not None

    @staticmethod
    def _create_storage_adapter(storage_type: StorageType) -> StorageAdapter:
        if storage_type == StorageType.LOCAL_FS:
            return LocalFileSystemAdapter()
        if storage_type == StorageType.GCS:
            return GoogleCloudStorageAdapter()
        if storage_type == StorageType.S3:
            raise NotImplementedError()
        if storage_type == StorageType.R2:
            raise NotImplementedError()

        raise NotImplementedError()

    def _get_storage_adapter(self, path: str) -> StorageAdapter:
        storage_type: Optional[StorageType] = None
        if StorageCleaner._is_url(path):
            parsed = urlparse(str(path))
            if parsed.scheme == "gs":
                storage_type = StorageType.GCS
            elif parsed.scheme == "s3":
                storage_type = StorageType.S3
            elif parsed.scheme == "r2":
                storage_type = StorageType.R2
            elif parsed.scheme == "file":
                path = path.replace("file://", "", 1)
                storage_type = StorageType.LOCAL_FS
        else:
            storage_type = StorageType.LOCAL_FS

        if storage_type is None:
            raise ValueError(f"Cannot determine storage type for path {path}")

        if storage_type not in self._storage_adapters:
            self._storage_adapters[storage_type] = StorageCleaner._create_storage_adapter(storage_type)

        return self._storage_adapters[storage_type]

    @staticmethod
    def _contains_config_yaml(dir_entries: List[str]) -> bool:
        return any(CONFIG_YAML.lower() == entry.lower() for entry in dir_entries)

    @staticmethod
    def _contains_nontrivial_checkpoint_dir(dir_entries: List[str]) -> bool:
        return any("step" in entry.lower() and "step0" not in entry.lower() for entry in dir_entries)

    def _verify_deletion_without_config_yaml(self, run_dir_entry: str):
        msg = f"No {CONFIG_YAML} found in run directory entry {run_dir_entry}. This entry might not correspond to a run."
        if self._runs_require_config_yaml:
            raise ValueError(msg)

        log.warning(msg)

        if not self._ignore_prompts:
            while True:
                response = input(f"{msg} Would you still like to delete {run_dir_entry}? (y/n) ")
                if response.lower() == "y":
                    break
                else:
                    raise ValueError(msg)

    def _delete_if_bad_run(self, storage: StorageAdapter, run_dir_entry: str):
        dir_entries = storage.list_entries(run_dir_entry)

        if not self._contains_config_yaml(dir_entries):
            self._verify_deletion_without_config_yaml(run_dir_entry)

        if not self._contains_nontrivial_checkpoint_dir(dir_entries):
            if self._dry_run:
                log.info("Would delete run_dir_entry %s", run_dir_entry)
            else:
                log.info("Deleting run_dir_entry %s", run_dir_entry)
                storage.delete_path(run_dir_entry)

    def delete_bad_runs(self, runs_path: str):
        log.info("Starting deletion of bad runs")

        if not runs_path.endswith("/"):
            raise ValueError(
                "Runs path does not end with '/'. Please verify that path is a directory and re-run with trailing '/'."
            )

        storage: StorageAdapter = self._get_storage_adapter(runs_path)
        run_dir_entries = [
            os.path.join(runs_path, entry)
            for entry in storage.list_entries(runs_path, max_file_size=self._max_archive_size)
        ]
        for run_dir_entry in run_dir_entries:
            self._delete_if_bad_run(storage, run_dir_entry)

    def _get_wandb_id(self, storage: StorageAdapter, run_dir_entry: str) -> str:
        dir_entries = storage.list_entries(run_dir_entry)
        if CONFIG_YAML not in dir_entries:
            raise FileNotFoundError(f'{CONFIG_YAML} not found in dir {run_dir_entry}, cannot get wandb id')

        config_yaml_path = os.path.join(run_dir_entry, CONFIG_YAML)
        train_config = TrainConfig.load(config_yaml_path)
        if train_config.wandb is None:
            raise ValueError(f'No wandb settings in config file {config_yaml_path}')

        entity_name = train_config.wandb.entity or self._default_wandb_entity
        project_name = train_config.wandb.project or self._default_wandb_project
        run_name = train_config.wandb.name

        if entity_name is None:
            raise ValueError(f'No wandb entity set in cli or in config file {config_yaml_path}')
        if project_name is None:
            raise ValueError(f'No wandb project name set in cli or in config file {config_yaml_path}')
        if run_name is None:
            raise ValueError(f'No wandb name set in config file {config_yaml_path}')

        wandb_api = wandb.Api()
        runs = list(wandb_api.runs(path=f'{entity_name}/{project_name}', filters={"display_name": {"$regex": run_name}}))
        if len(runs) == 0:
            raise ValueError(f'No wandb runs found for {run_dir_entry}')
        if len(runs) > 1:
            raise ValueError(f'{len(runs)} runs found for {run_dir_entry}')

        run = runs[0]
        print('id', run.id)
        return run.id

    def rename_runs_to_wandb_ids(self, runs_path: str):
        log.info("Starting renaming runs to their wandb ids")

        if not runs_path.endswith("/"):
            raise ValueError(
                "Runs path does not end with '/'. Please verify that path is a directory and re-run with trailing '/'."
            )

        storage: StorageAdapter = self._get_storage_adapter(runs_path)
        run_dir_entries = [
            os.path.join(runs_path, entry)
            for entry in storage.list_dirs(runs_path)
        ]

        print(run_dir_entries)
        run_wandb_ids = {
            run_dir_entry: self._get_wandb_id(storage, run_dir_entry)
            for run_dir_entry in run_dir_entries
        }
        print(run_wandb_ids)

        raise NotImplementedError


def perform_operation(args: argparse.Namespace):
    if args.dry_run:
        log.info("Dry run, no actions will be taken")

    if args.op == CleaningOperations.DELETE_BAD_RUNS:
        storage_manager = StorageCleaner(
            dry_run=args.dry_run,
            ignore_prompts=args.yes,
            runs_require_config_yaml=args.runs_require_config_yaml,
            max_archive_size=args.max_archive_size,
        )
        storage_manager.delete_bad_runs(args.runs_path)
    if args.op == CleaningOperations.RENAME_RUNS_TO_WANDB_ID:
        storage_manager = StorageCleaner(
            dry_run=args.dry_run,
            ignore_prompts=args.yes,
            default_wandb_entity=args.entity,
            default_wandb_project=args.project,
        )
        storage_manager.rename_runs_to_wandb_ids(args.runs_path)


def _add_delete_subparser(subparsers: _SubParsersAction):
    delete_runs_parser = subparsers.add_parser(
        "clean", help="Delete bad runs (example no non-trivial checkpoints)"
    )
    delete_runs_parser.set_defaults(op=CleaningOperations.DELETE_BAD_RUNS)
    delete_runs_parser.add_argument(
        "runs_path",
        help="Path to directory containing one or more run directories",
    )
    delete_runs_parser.add_argument(
        "--require_config_yaml",
        action="store_true",
        dest="runs_require_config_yaml",
        help=f"Enforces without prompt the sanity check that an entry being deleted has a {CONFIG_YAML} file (and so is a run)",
    )
    delete_runs_parser.add_argument(
        "--max_archive_size",
        default=DEFAULT_MAX_ARCHIVE_SIZE,
        help="Max size archive files to consider for deletion (in bytes). Any archive larger than this is ignored/not deleted.",
    )


def _add_wandb_subparser(subparsers: _SubParsersAction):
    wandb_runs_parser = subparsers.add_parser(
        "rename_to_wandb", help="renames runs to their wandb ids"
    )
    wandb_runs_parser.set_defaults(op=CleaningOperations.RENAME_RUNS_TO_WANDB_ID)
    wandb_runs_parser.add_argument(
        "runs_path",
        help="Path to directory containing one or more run directories",
    )
    wandb_runs_parser.add_argument(
        "--entity",
        default=WANDB_ENTITY,
        help="Wandb entity to use for runs without a specified entity.",
    )
    wandb_runs_parser.add_argument(
        "--project",
        default=None,
        help="Wandb project to use for runs without a specified project. If unset, runs without a specified project will be skipped.",
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--dry_run",
        action="store_true",
        help="If set, indicate actions but do not do them",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="If set, bypass prompts",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        help="Sets the logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Cleaning commands")
    _add_delete_subparser(subparsers)
    _add_wandb_subparser(subparsers)

    # gs://ai2-olmo/ai2-llm/olmo-medium/njmmt4v8/config.yaml
    # temp
    # gs://ai2-olmo/unsorted-checkpoints/3416090.tar.bz2

    return parser


def main():
    args = get_parser().parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    perform_operation(args)


if __name__ == "__main__":
    main()
