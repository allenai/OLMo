import argparse
import logging
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from argparse import ArgumentParser, _SubParsersAction
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3.session
import botocore.exceptions as boto_exceptions
import google.cloud.storage as gcs
import torch
import wandb
from boto3.s3.transfer import TransferConfig
from cached_path import add_scheme_client, cached_path, set_cache_dir
from cached_path.schemes import S3Client
from google.api_core.exceptions import NotFound
from omegaconf import OmegaConf as om
from rich.progress import track

from olmo import util
from olmo.aliases import PathOrStr
from olmo.checkpoint import (
    Checkpointer,
    LocalShardedCheckpointer,
    TorchLegacyShardedCheckpointer,
)
from olmo.config import ShardedCheckpointerType, TrainConfig

log = logging.getLogger(__name__)


CONFIG_YAML: str = "config.yaml"
DEFAULT_DELETE_MAX_ARCHIVE_SIZE: float = 5 * 1024 * 1024 * 1024  # 5GB


class CleaningOperations(Enum):
    DELETE_BAD_RUNS = auto()
    UNSHARD_CHECKPOINTS = auto()
    MOVE_RUN = auto()


class StorageType(util.StrEnum):
    LOCAL_FS = ""
    GCS = "gs"
    S3 = "s3"
    R2 = "r2"


class StorageAdapter(ABC):
    @abstractmethod
    def list_entries(self, directory: str, max_file_size: Optional[int] = None) -> List[str]:
        """Lists all the entries within the given directory.
        Returns only top-level entries (i.e. not entries in subdirectories).

        `max_file_size`: Sets a threshold (in bytes) for the largest size file to retain within entries.
        Any file of larger size is not included in the returned results.
        """

    @abstractmethod
    def list_dirs(self, directory: str) -> List[str]:
        """Lists all the directories within the given directory.
        Returns only top-level entries (i.e. not entries in subdirectories).
        """

    @abstractmethod
    def delete_path(self, path: str):
        """Deletes the entry at the given path and, if the path is a directory, delete all entries
        within its subdirectories.
        """

    @abstractmethod
    def is_file(self, path: str) -> bool:
        """Returns whether the given path corresponds to an existing file."""

    @abstractmethod
    def get_file_size(self, path: str) -> int:
        """Get the size of the file at the given path in bytes. Raises an error if the path does not
        correspond to a file.
        """

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        """Returns whether the given path corresponds to an existing directory."""

    @abstractmethod
    def download_folder(self, directory_path: str, local_dest_folder: PathOrStr):
        """Downloads the content from the directory path to the local FS destination folder."""

    @abstractmethod
    def upload(self, local_src: PathOrStr, dest_path: str):
        """Uploads the content from the directory or file at the local FS source to the path."""

    @classmethod
    def create_storage_adapter(cls, storage_type: StorageType):
        if storage_type == StorageType.LOCAL_FS:
            return LocalFileSystemAdapter()
        if storage_type == StorageType.GCS:
            return GoogleCloudStorageAdapter()
        if storage_type in (StorageType.S3, StorageType.R2):
            return S3StorageAdapter(storage_type)

        raise NotImplementedError(f"No storage adapter implemented for storage type {storage_type}")

    @staticmethod
    def _is_url(path: str) -> bool:
        return re.match(r"[a-z0-9]+://.*", path) is not None

    @staticmethod
    def get_storage_type_for_path(path: str) -> StorageType:
        if StorageAdapter._is_url(path):
            parsed = urlparse(str(path))
            if parsed.scheme == "gs":
                return StorageType.GCS
            elif parsed.scheme == "s3":
                return StorageType.S3
            elif parsed.scheme == "r2":
                return StorageType.R2
            elif parsed.scheme == "file":
                return StorageType.LOCAL_FS

        return StorageType.LOCAL_FS


class LocalFileSystemAdapter(StorageAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._temp_files: List[tempfile._TemporaryFileWrapper[bytes]] = []
        self._temp_dirs: List[tempfile.TemporaryDirectory] = []
        self._archive_extensions: List[str] = [
            extension.lower() for _, extensions, _ in shutil.get_unpack_formats() for extension in extensions
        ]

    def __del__(self):
        for temp_file in self._temp_files:
            temp_file.close()
        for temp_dir in self._temp_dirs:
            temp_dir.cleanup()

    def create_temp_file(self, suffix: Optional[str] = None) -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
        self._temp_files.append(temp_file)
        return temp_file.name

    def create_temp_dir(self, directory: Optional[str] = None, suffix: Optional[str] = None) -> str:
        temp_dir = tempfile.TemporaryDirectory(dir=directory, suffix=suffix)
        self._temp_dirs.append(temp_dir)
        return temp_dir.name

    def has_supported_archive_extension(self, path: PathOrStr) -> bool:
        filename = Path(path).name.lower()
        return any(filename.endswith(extension) for extension in self._archive_extensions)

    def _list_entries(
        self, directory: PathOrStr, include_files: bool = True, max_file_size: Optional[int] = None
    ) -> List[str]:
        dir_obj = Path(directory)
        if not dir_obj.is_dir():
            raise ValueError(f"{directory} is not an existing directory")

        return [
            entry.name
            for entry in dir_obj.iterdir()
            if (
                (include_files or not entry.is_file())
                and (not entry.is_file() or max_file_size is None or self._get_file_size(entry) <= max_file_size)
            )
        ]

    def list_entries(self, directory: str, max_file_size: Optional[int] = None) -> List[str]:
        return self._list_entries(directory, max_file_size=max_file_size)

    def list_dirs(self, directory: str) -> List[str]:
        return self._list_entries(directory, include_files=False)

    def delete_path(self, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            return

        if path_obj.is_file():
            path_obj.unlink()
        else:
            shutil.rmtree(path)

    def is_file(self, path: str) -> bool:
        return Path(path).is_file()

    def _get_file_size(self, path: Path) -> int:
        if not path.is_file():
            raise ValueError(f"Path does not correspond to an existing file: {path}")

        return path.stat().st_size

    def get_file_size(self, path: str) -> int:
        return self._get_file_size(Path(path))

    def is_dir(self, path: str) -> bool:
        path_obj = Path(path)
        if not path_obj.exists():
            return False

        return path_obj.is_dir()

    def download_folder(self, directory_path: str, local_dest_folder: PathOrStr):
        directory_path_obj = Path(directory_path)
        if not directory_path_obj.exists():
            raise ValueError(f"No entry exists at path {directory_path}")

        if directory_path_obj.is_dir():
            shutil.copytree(directory_path, str(local_dest_folder), dirs_exist_ok=True)
        else:
            raise RuntimeError(f"Unexpected type of path {directory_path}")

    def upload(self, local_src: PathOrStr, dest_path: str):
        local_src_obj = Path(local_src)
        if local_src_obj.is_file():
            shutil.copy(str(local_src_obj), dest_path)
        elif local_src_obj.is_dir():
            self.download_folder(str(local_src), dest_path)
        else:
            raise RuntimeError(f"Unexpected type of local src path {local_src}")


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

    @staticmethod
    def _get_path(bucket_name: str, key: str) -> str:
        return f"gs://{bucket_name}/{key}"

    def _get_blob_size(self, blob: gcs.Blob) -> int:
        blob.reload()
        if blob.size is None:
            raise ValueError(f"Failed to get size for blob: {blob.name}")
        return blob.size

    def _is_file(self, bucket_name: str, key: str) -> bool:
        if len(key) == 0:
            return False

        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(key)
        try:
            blob.reload()
            return True
        except NotFound:
            return False

    def _get_size(self, bucket_name: str, key: str) -> int:
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.get_blob(key)
        if blob is None:
            raise ValueError(f"Getting size for invalid object: {self._get_path(bucket_name, key)}")

        return self._get_blob_size(blob)

    def _get_directory_entries(
        self,
        bucket_name: str,
        key: str,
        include_files: bool = True,
        max_file_size: Optional[int] = None,
    ) -> List[str]:
        bucket = self.gcs_client.bucket(bucket_name)
        # Using delimiter causes result to have directory-like structure
        blobs = bucket.list_blobs(prefix=key, delimiter="/")

        entries: List[str] = []
        for blob in blobs:
            if not include_files:
                # Note: We need to iterate through (or otherwise act on?) the blobs to populate blob.prefixes
                # Thus we no-op here rather than skipping the loop
                continue

            size: int = self._get_blob_size(blob)
            if max_file_size is not None and size > max_file_size:
                log.info(
                    "Blob %s has size %.2fGb exceeding max file size %.2fGb, skipping.",
                    blob.name,
                    size / (1024 * 1024 * 1024),
                    max_file_size / (1024 * 1024 * 1024),
                )
                continue

            entries.append(blob.name)  # type: ignore

        # Note: We need to iterate through (or otherwise act on?) the blobs to populate blob.prefixes
        entries += blobs.prefixes

        return [entry.removeprefix(key) for entry in entries]

    def _list_entries(
        self, directory: str, include_files: bool = True, max_file_size: Optional[int] = None
    ) -> List[str]:
        bucket_name, key = self._get_bucket_name_and_key(directory)

        if not self._is_dir(bucket_name, key):
            raise ValueError(f"{directory} is not an existing directory")

        res = self._get_directory_entries(
            bucket_name, key, include_files=include_files, max_file_size=max_file_size
        )
        return res

    def list_entries(self, directory: str, max_file_size: Optional[int] = None) -> List[str]:
        return self._list_entries(directory, max_file_size=max_file_size)

    def list_dirs(self, directory: str) -> List[str]:
        return self._list_entries(directory, include_files=False)

    def delete_path(self, path: str):
        bucket_name, key = self._get_bucket_name_and_key(path)

        bucket = self.gcs_client.bucket(bucket_name)
        # Not using delimiter causes result to not have directory-like structure (all blobs returned)
        blobs = list(bucket.list_blobs(prefix=key))

        bucket.delete_blobs(blobs)

    def is_file(self, path: str) -> bool:
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._is_file(bucket_name, key)

    def get_file_size(self, path: str) -> int:
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._get_size(bucket_name, key)

    def _is_dir(self, bucket_name: str, key: str) -> bool:
        key = f"{key}/" if not key.endswith("/") else key

        bucket = self.gcs_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=key, max_results=1))

        return not self._is_file(bucket_name, key) and len(blobs) > 0

    def is_dir(self, path: str) -> bool:
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._is_dir(bucket_name, key)

    def download_folder(self, directory_path: str, local_dest_folder: PathOrStr):
        bucket_name, key = self._get_bucket_name_and_key(directory_path)
        bucket = self.gcs_client.bucket(bucket_name)

        if self._is_dir(bucket_name, key):
            blobs: List[gcs.Blob] = list(bucket.list_blobs(prefix=key))

            for blob in track(blobs, description=f"Downloading files at {directory_path}"):
                if not blob.name:
                    raise NotImplementedError()
                blob_path: str = blob.name
                blob_local_dest = blob_path.replace(key.rstrip("/"), str(local_dest_folder).rstrip("/"))
                blob.download_to_filename(blob_local_dest)
        else:
            raise ValueError(f"Path {directory_path} is not a valid directory")

    def upload(self, local_src: PathOrStr, dest_path: str):
        raise NotImplementedError()


class S3StorageAdapter(StorageAdapter):
    def __init__(self, storage_type: StorageType):
        super().__init__()
        self._storage_type = storage_type
        self._s3_client = util._get_s3_client(str(storage_type))

        self._local_fs_adapter: Optional[LocalFileSystemAdapter] = None
        self._temp_dirs: List[tempfile.TemporaryDirectory] = []

    @property
    def local_fs_adapter(self):
        if self._local_fs_adapter is None:
            self._local_fs_adapter = LocalFileSystemAdapter()

        return self._local_fs_adapter

    @staticmethod
    def _get_bucket_name_and_key(path: str) -> Tuple[str, str]:
        parsed_path = urlparse(path)
        bucket_name = parsed_path.netloc
        key = parsed_path.path.lstrip("/")
        return bucket_name, key

    def _get_path(self, bucket_name: str, key: str) -> str:
        scheme: str
        if self._storage_type == StorageType.S3:
            scheme = "s3"
        elif self._storage_type == StorageType.R2:
            scheme = "r2"
        else:
            raise NotImplementedError

        return f"{scheme}://{bucket_name}/{key}"

    def _get_size(self, bucket_name: str, key: str) -> int:
        if not self._is_file(bucket_name, key):
            raise ValueError(f"Provided path does not correspond to a file: {self._get_path(bucket_name, key)}")

        head_response: Dict[str, Any] = self._s3_client.head_object(Bucket=bucket_name, Key=key)
        if "ContentLength" not in head_response:
            raise RuntimeError(f"Failed to get size for file: {self._get_path(bucket_name, key)}")
        return head_response["ContentLength"]

    def _get_directory_entries(
        self,
        bucket_name: str,
        key: str,
        include_files: bool = True,
        max_file_size: Optional[int] = None,
    ) -> List[str]:
        response: Dict[str, Any] = self._s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key, Delimiter="/")

        entries: List[str] = []

        if include_files:
            objects_metadata: List[Dict[str, Any]] = response.get("Contents", [])
            for object_metadata in objects_metadata:
                object_name = object_metadata["Key"]

                size: int = object_metadata["Size"]
                if max_file_size is not None and size > max_file_size:
                    log.info(
                        "Object %s has size %.2fGiB exceeding max file size %.2fGiB, skipping.",
                        object_name,
                        size / (1024 * 1024 * 1024),
                        max_file_size / (1024 * 1024 * 1024),
                    )
                    continue

                entries.append(object_name)

        directories_metadata: List[Dict[str, str]] = response.get("CommonPrefixes", [])
        entries += [directory_metadata["Prefix"] for directory_metadata in directories_metadata]

        return [entry.removeprefix(key) for entry in entries]

    def _list_entries(
        self, directory: str, include_files: bool = True, max_file_size: Optional[int] = None
    ) -> List[str]:
        bucket_name, key = self._get_bucket_name_and_key(directory)

        if not self._is_dir(bucket_name, key):
            raise ValueError(f"{directory} is not an existing directory")

        res = self._get_directory_entries(
            bucket_name, key, include_files=include_files, max_file_size=max_file_size
        )
        return res

    def list_entries(self, directory: str, max_file_size: Optional[int] = None) -> List[str]:
        return self._list_entries(directory, max_file_size=max_file_size)

    def list_dirs(self, directory: str) -> List[str]:
        return self._list_entries(directory, include_files=False)

    def delete_path(self, path: str):
        bucket_name, key = self._get_bucket_name_and_key(path)

        response: Dict[str, Any] = self._s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key)

        objects_metadata: List[Dict[str, Any]] = response.get("Contents", [])
        object_keys_to_delete: List[str] = [object_metadata["Key"] for object_metadata in objects_metadata]

        log.info("Starting to delete %d objects at %s", len(object_keys_to_delete), path)

        max_delete_batch_size: int = 1000
        for i in range(0, len(object_keys_to_delete), max_delete_batch_size):
            delete_batch_keys = [
                {"Key": object_key} for object_key in object_keys_to_delete[i : i + max_delete_batch_size]
            ]

            delete_response: Dict[str, Any] = self._s3_client.delete_objects(
                Bucket=bucket_name, Delete={"Objects": delete_batch_keys}
            )

            errors: List[Dict[str, Any]] = delete_response.get("Errors", [])
            if len(errors) > 0:
                for error in errors:
                    log.error(
                        "Failed to delete %s with code %s, message %s",
                        error["Key"],
                        error["Code"],
                        error["Message"],
                    )

                raise RuntimeError(f"Error occurred during deletion at {path}")

            deleted_object_keys = [deleted_object["Key"] for deleted_object in delete_response.get("Deleted", [])]
            delete_batch_keys_set = set(object_keys_to_delete[i : i + max_delete_batch_size])
            deleted_object_keys_set = set(deleted_object_keys)
            unrequested_deleted_keys = deleted_object_keys_set.difference(delete_batch_keys_set)
            if len(unrequested_deleted_keys) > 0:
                raise RuntimeError(f"The following keys were unexpectedly deleted: {unrequested_deleted_keys}")
            undeleted_keys = delete_batch_keys_set.difference(deleted_object_keys_set)
            if len(undeleted_keys) > 0:
                raise RuntimeError(f"The following keys failed to be deleted: {undeleted_keys}")

    def _is_file(self, bucket_name: str, key: str) -> bool:
        if len(key) == 0:
            return False

        try:
            self._s3_client.head_object(Bucket=bucket_name, Key=key)
            return True
        except boto_exceptions.ClientError as e:
            if int(e.response["Error"]["Code"]) == 404:
                return False

            raise e

    def is_file(self, path: str) -> bool:
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._is_file(bucket_name, key)

    def get_file_size(self, path: str) -> int:
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._get_size(bucket_name, key)

    def _is_dir(self, bucket_name: str, key: str) -> bool:
        key = f"{key}/" if not key.endswith("/") else key
        if self._is_file(bucket_name, key):
            return False

        response = self._s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key, MaxKeys=1)
        return "Contents" in response

    def is_dir(self, path: str) -> bool:
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._is_dir(bucket_name, key)

    def download_folder(self, directory_path: str, local_dest_folder: PathOrStr):
        bucket_name, key = self._get_bucket_name_and_key(directory_path)

        if self._is_dir(bucket_name, key):
            response = self._s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key)
            objects_metadata: List[Dict[str, Any]] = response["Contents"]
            for object_metadata in track(objects_metadata, description=f"Downloading files at {directory_path}"):
                object_key: str = object_metadata["Key"]
                object_local_dest = object_key.replace(key.rstrip("/"), str(local_dest_folder).rstrip("/"))

                self._s3_client.download_file(bucket_name, object_key, object_local_dest)
        else:
            raise ValueError(f"Path {directory_path} is not a valid directory")

    def _upload_file(self, local_filepath: str, bucket_name: str, key: str):
        transfer_config = TransferConfig(max_concurrency=4)
        self._s3_client.upload_file(local_filepath, bucket_name, key, Config=transfer_config)

    def upload(self, local_src: PathOrStr, dest_path: str):
        if self.local_fs_adapter.is_file(str(local_src)):
            bucket_name, key = self._get_bucket_name_and_key(dest_path)
            self._upload_file(str(local_src), bucket_name, key)

        elif self.local_fs_adapter.is_dir(str(local_src)):
            local_src = Path(local_src)

            local_file_paths = list(local_src.rglob("*"))
            for file_local_path in track(local_file_paths, description=f"Uploading to {dest_path}"):
                if file_local_path.is_dir():
                    continue

                file_dest_path = str(file_local_path).replace(str(local_src).rstrip("/"), dest_path.rstrip("/"))
                bucket_name, key = self._get_bucket_name_and_key(file_dest_path)

                if not self._is_file(bucket_name, key):
                    self._upload_file(str(file_local_path), bucket_name, key)

        else:
            raise ValueError(f"Local source {local_src} does not correspond to a valid file or directory")


@dataclass
class StorageCleanerConfig:
    dry_run: bool
    temp_dir: str


@dataclass
class DeleteBadRunsConfig(StorageCleanerConfig):
    should_check_is_run: bool
    ignore_non_runs: bool
    max_archive_size: Optional[int]


@dataclass
class UnshardCheckpointsConfig(StorageCleanerConfig):
    latest_checkpoint_only: bool
    delete_sharded_checkpoints: bool
    checkpoint_num: Optional[int]


@dataclass
class MoveRunConfig(StorageCleanerConfig):
    append_wandb_path: bool
    keep_src: bool
    store_archived: bool


def _get_storage_adapter_for_path(path: str) -> StorageAdapter:
    storage_type = StorageAdapter.get_storage_type_for_path(path)
    return StorageAdapter.create_storage_adapter(storage_type)


def _contains_checkpoint_dir(dir_entries: List[str]) -> bool:
    return any(re.match(r"step\d+(-unsharded)?", entry) is not None for entry in dir_entries)


def _contains_nontrivial_checkpoint_dir(dir_entries: List[str]) -> bool:
    return any(re.match(r"step[1-9]\d*(-unsharded)?", entry) is not None for entry in dir_entries)


def _is_run(directory: str, run_entries: Optional[List[str]] = None) -> bool:
    """
    This method is best effort. It may mark run paths as not (false negatives) or mark non-run
    paths as runs (false positives). We prioritize minimizing false positives.
    """
    storage = _get_storage_adapter_for_path(directory)
    if run_entries is None:
        run_entries = storage.list_entries(directory)

    if CONFIG_YAML in run_entries:
        # A directory with both config.yaml and a wandb subdirectory is most likely a run
        if storage.is_dir(os.path.join(directory, "wandb")):
            return True

        # A directory with both config.yaml and a train_data subdirectory is most likely a run
        if storage.is_dir(os.path.join(directory, "train_data")):
            return True

    return _contains_checkpoint_dir(run_entries)


def _verify_non_run_deletion(run_dir_or_archive: str, run_entries: List[str], config: DeleteBadRunsConfig):
    msg = f"Attempting to delete non-run directory or archive {run_dir_or_archive} (first 5 entries: {run_entries[:5]})."
    if config.ignore_non_runs:
        log.warning(msg)
        return

    raise ValueError(msg)


def _is_archive(path: str, storage: StorageAdapter) -> bool:
    local_storage = LocalFileSystemAdapter()
    return local_storage.has_supported_archive_extension(path) and storage.is_file(path)


def _format_dir_or_archive_path(storage: StorageAdapter, path: str) -> str:
    if storage.is_dir(path):
        return f"{path}/" if not path.endswith("/") else path

    if _is_archive(path, storage):
        return path

    raise ValueError(f"Path does not correspond to a directory or archive file: {path}")


def _unarchive_if_archive(dir_or_archive: str, storage: StorageAdapter) -> str:
    if _is_archive(dir_or_archive, storage):
        unarchived_dir = cached_path(dir_or_archive, extract_archive=True)
        assert unarchived_dir != Path(dir_or_archive)

        # The unarchived file could have a redundant top-level directory. If the top-level
        # directory has only a directory, we should return that directory instead.
        unarchived_dir_storage = _get_storage_adapter_for_path(str(unarchived_dir))
        unarchived_dir_entries = unarchived_dir_storage.list_entries(str(unarchived_dir))
        if len(unarchived_dir_entries) == 1:
            unarchived_entry_path = unarchived_dir / unarchived_dir_entries[0]
            if unarchived_dir_storage.is_dir(str(unarchived_entry_path)):
                return str(unarchived_entry_path)

        return str(unarchived_dir)

    if storage.is_dir(dir_or_archive):
        return dir_or_archive

    raise ValueError(f"Dir or archive {dir_or_archive} is not a valid archive file or directory")


def _should_delete_run(storage: StorageAdapter, run_dir_or_archive: str, config: DeleteBadRunsConfig) -> bool:
    # Do not delete archive files that are bigger than the configured max
    if config.max_archive_size is not None and storage.is_file(run_dir_or_archive):
        file_size = storage.get_file_size(run_dir_or_archive)
        if file_size > config.max_archive_size:
            log.info(
                "File size %d of %s exceeds max archive size %s",
                file_size,
                run_dir_or_archive,
                config.max_archive_size,
            )
            return False

    run_dir = _unarchive_if_archive(run_dir_or_archive, storage)
    run_dir_storage = _get_storage_adapter_for_path(run_dir)

    run_entries = run_dir_storage.list_entries(run_dir)
    if config.should_check_is_run and not _is_run(run_dir, run_entries=run_entries):
        _verify_non_run_deletion(run_dir_or_archive, run_entries, config)
        return False

    # Runs with non-trivial checkpoints are considered good
    if _contains_nontrivial_checkpoint_dir(run_entries):
        log.info("Run directory or archive %s contains a non-trivial checkpoint directory", run_dir_or_archive)
        return False

    return True


def _delete_if_bad_run(storage: StorageAdapter, run_path: str, config: DeleteBadRunsConfig):
    run_dir_or_archive = _format_dir_or_archive_path(storage, run_path)

    if _should_delete_run(storage, run_dir_or_archive, config):
        if config.dry_run:
            log.info("Would delete run directory or archive %s", run_dir_or_archive)
        else:
            log.info("Deleting run directory or archive %s", run_dir_or_archive)
            storage.delete_path(run_dir_or_archive)
    else:
        log.info("Skipping run directory or archive %s", run_dir_or_archive)


def delete_bad_runs(run_paths: List[str], config: DeleteBadRunsConfig):
    for run_path in run_paths:
        storage: StorageAdapter = _get_storage_adapter_for_path(run_path)
        log.info("Starting to check if run %s should be deleted", run_path)
        _delete_if_bad_run(storage, run_path, config)

        # Delete temp dir after each run to avoid storage bloat
        if Path(config.temp_dir).is_dir():
            log.info("Deleting temp dir %s", config.temp_dir)
            shutil.rmtree(config.temp_dir)


def _is_sharded_checkpoint_dir(directory: str) -> bool:
    storage = _get_storage_adapter_for_path(directory)
    return storage.is_dir(directory) and re.match(r"step\d+$", Path(directory).name) is not None


def _get_checkpoint_number(checkpoint_dir: str) -> int:
    checkpoint_dir_name = Path(checkpoint_dir).name
    checkpoint_dir_name = checkpoint_dir_name.removesuffix("-unsharded")
    match = re.match(r"step(\d+)$", checkpoint_dir_name)
    if match is None:
        raise ValueError(f"Failed to find checkpoint number for dir {checkpoint_dir}")

    return int(match.group(1))


def _get_sharded_checkpoint_dirs(
    run_dir_storage: StorageAdapter, run_dir: str, run_dir_or_archive: str, latest_checkpoint_only: bool
) -> List[str]:
    run_subdir_names = run_dir_storage.list_dirs(run_dir)
    run_subdirectories = list(map(lambda dir_name: os.path.join(run_dir, dir_name), run_subdir_names))
    sharded_checkpoint_directories = list(filter(_is_sharded_checkpoint_dir, run_subdirectories))

    if latest_checkpoint_only:
        latest_checkpoint_directory = max(sharded_checkpoint_directories, default=None, key=_get_checkpoint_number)
        sharded_checkpoint_directories = (
            [latest_checkpoint_directory] if latest_checkpoint_directory is not None else []
        )

    log.info(
        "Found %d sharded checkpoint directories for %s", len(sharded_checkpoint_directories), run_dir_or_archive
    )

    return sharded_checkpoint_directories


def _add_training_config_to_checkpoint(local_checkpoint_dir: str, run_dir: str) -> bool:
    max_train_config_size = 1 * 1024 * 1024  # 1MB

    if not StorageAdapter.get_storage_type_for_path(local_checkpoint_dir) == StorageType.LOCAL_FS:
        raise ValueError(f"Checkpoint dir is not local: {local_checkpoint_dir}")

    checkpoint_storage = _get_storage_adapter_for_path(local_checkpoint_dir)
    if CONFIG_YAML in checkpoint_storage.list_entries(local_checkpoint_dir, max_file_size=max_train_config_size):
        # Config already exists in the checkpoint
        return False

    log.info("%s not found in %s, attempting to get it from %s", CONFIG_YAML, local_checkpoint_dir, run_dir)

    run_storage = _get_storage_adapter_for_path(run_dir)
    run_config_yaml_path = os.path.join(run_dir, CONFIG_YAML)
    if run_storage.is_file(run_config_yaml_path):
        local_config_yaml_path = cached_path(run_config_yaml_path)
        shutil.copy(local_config_yaml_path, local_checkpoint_dir)
        return True

    log.warning("Cannot find training config to add to checkpoint %s", local_checkpoint_dir)
    return False


def _unshard_checkpoint(
    sharded_checkpoint_dir: str, dest_dir: str, run_dir: str, unsharding_config: UnshardCheckpointsConfig
):
    local_storage = LocalFileSystemAdapter()

    # Download checkpoint to a temp dir if it is in cloud storage
    if StorageAdapter.get_storage_type_for_path(sharded_checkpoint_dir) != StorageType.LOCAL_FS:
        sharding_input_dir = local_storage.create_temp_dir(directory=unsharding_config.temp_dir)
        src_storage = _get_storage_adapter_for_path(sharded_checkpoint_dir)
        src_storage.download_folder(sharded_checkpoint_dir, sharding_input_dir)
    else:
        sharding_input_dir = sharded_checkpoint_dir

    training_config_added = _add_training_config_to_checkpoint(sharding_input_dir, run_dir)

    # Set unsharder output to a temp dir
    sharding_output_dir: str
    sharding_output_dir = local_storage.create_temp_dir(directory=unsharding_config.temp_dir)

    try:
        config = TrainConfig.load(Path(sharding_input_dir) / "config.yaml", validate_paths=False)
        sharded_checkpoint_type = config.sharded_checkpointer
        checkpointer: Checkpointer
        if sharded_checkpoint_type == ShardedCheckpointerType.torch_legacy:
            checkpointer = TorchLegacyShardedCheckpointer(config)
        elif sharded_checkpoint_type == ShardedCheckpointerType.local:
            checkpointer = LocalShardedCheckpointer(config)
        else:
            raise NotImplementedError(sharded_checkpoint_type)

        model_state_dict, optim_state_dict, trainer_state_dict = checkpointer.unshard_checkpoint(
            sharding_input_dir
        )
    except RuntimeError as e:
        log.error(
            "Unsharding from %s to %s failed with exception: %s",
            sharding_input_dir,
            sharding_output_dir,
            e,
        )

        if training_config_added:
            local_storage.delete_path(str(Path(sharding_input_dir) / CONFIG_YAML))

        local_storage.delete_path(sharding_output_dir)
        return

    # model
    model_output = str(Path(sharding_output_dir) / "model.pt")
    log.info("Saving model state to %s", model_output)
    torch.save(model_state_dict, model_output)
    del model_state_dict

    # optimizer
    optim_output = str(Path(sharding_output_dir) / "optim.pt")
    log.info("Saving optimizer state to %s", optim_output)
    torch.save(optim_state_dict, optim_output)
    del optim_state_dict

    # trainer
    train_output = str(Path(sharding_output_dir) / "train.pt")
    log.info("Saving everything else to %s", train_output)
    torch.save(trainer_state_dict, train_output)
    del trainer_state_dict

    log.info("Copying config.yaml to %s", sharding_output_dir)
    shutil.copy(Path(sharding_input_dir) / "config.yaml", sharding_output_dir)

    log.info(
        "Successfully unsharded from %s to %s, starting upload to %s",
        sharding_input_dir,
        sharding_output_dir,
        dest_dir,
    )

    dest_storage = _get_storage_adapter_for_path(dest_dir)
    dest_storage.upload(sharding_output_dir, dest_dir)


def _unshard_checkpoints(
    run_storage: StorageAdapter,
    run_dir_or_archive: str,
    checkpoints_dest_dir: str,
    config: UnshardCheckpointsConfig,
):
    log.info("Starting unsharding checkpoints of run directory or archive %s", run_dir_or_archive)

    run_dir = _unarchive_if_archive(run_dir_or_archive, run_storage)
    run_dir_storage = _get_storage_adapter_for_path(run_dir)

    sharded_checkpoint_directories = _get_sharded_checkpoint_dirs(
        run_dir_storage, run_dir, run_dir_or_archive, config.latest_checkpoint_only
    )
    for sharded_checkpoint_directory in sharded_checkpoint_directories:
        sharded_checkpoint_dir_name = Path(sharded_checkpoint_directory).name

        unsharded_checkpoint_directory_in_source = os.path.join(
            run_dir, f"{sharded_checkpoint_dir_name}-unsharded"
        )
        if run_dir_storage.is_dir(unsharded_checkpoint_directory_in_source):
            log.info(
                "Unsharded directory already exists for %s at source %s, skipping",
                sharded_checkpoint_dir_name,
                unsharded_checkpoint_directory_in_source,
            )
            continue

        dest_directory = os.path.join(checkpoints_dest_dir, f"{sharded_checkpoint_dir_name}-unsharded")
        dest_storage = _get_storage_adapter_for_path(dest_directory)
        if dest_storage.is_dir(dest_directory):
            log.info(
                "Unsharded directory already exists for %s at destination %s, skipping",
                sharded_checkpoint_dir_name,
                dest_directory,
            )
            continue

        if config.dry_run:
            log.info("Would unshard sharded checkpoint %s to %s", sharded_checkpoint_directory, dest_directory)
        else:
            log.info("Unsharding sharded checkpoint %s to %s", sharded_checkpoint_directory, dest_directory)
            _unshard_checkpoint(sharded_checkpoint_directory, dest_directory, run_dir, config)


def unshard_run_checkpoints(run_path: str, checkpoints_dest_dir: str, config: UnshardCheckpointsConfig):
    storage = _get_storage_adapter_for_path(run_path)
    run_dir_or_archive = _format_dir_or_archive_path(storage, run_path)
    _unshard_checkpoints(storage, run_dir_or_archive, checkpoints_dest_dir, config)


def _get_wandb_runs_from_wandb_dir(storage: StorageAdapter, wandb_dir: str, run_config: TrainConfig) -> List:
    # For some reason, we often have a redundant nested wandb directory. Step into it here.
    nested_wandb_dir = os.path.join(wandb_dir, "wandb/")
    if storage.is_dir(nested_wandb_dir):
        wandb_dir = nested_wandb_dir

    # Wandb run directory names are stored in format <run>-<timestamp>-<id>
    # https://docs.wandb.ai/guides/track/save-restore#examples-of-wandbsave
    dir_names = storage.list_dirs(wandb_dir)
    wandb_run_dir_names = [dir_name for dir_name in dir_names if dir_name.startswith("run")]
    if len(wandb_run_dir_names) == 0:
        log.warning("No wandb run directories found in wandb dir %s", wandb_dir)
        return []

    wandb_ids = [dir_name.split("-")[2] for dir_name in wandb_run_dir_names if dir_name.count("-") >= 2]

    log.debug("Wandb ids: %s", wandb_ids)

    assert run_config.wandb is not None
    api: wandb.Api = wandb.Api()
    return [api.run(path=f"{run_config.wandb.entity}/{run_config.wandb.project}/{id}") for id in wandb_ids]


def _get_wandb_path_from_run(wandb_run) -> str:
    return "/".join(wandb_run.path)


def _get_wandb_runs_from_train_config(config: TrainConfig) -> List:
    assert config.wandb is not None

    run_filters = {
        "display_name": config.wandb.name,
    }
    if config.wandb.group is not None:
        run_filters["group"] = config.wandb.group

    log.debug("Wandb entity/project: %s/%s", config.wandb.entity, config.wandb.project)
    log.debug("Wandb filters: %s", run_filters)

    api = wandb.Api()
    return api.runs(path=f"{config.wandb.entity}/{config.wandb.project}", filters=run_filters)


def _are_equal_configs(wandb_config: TrainConfig, train_config: TrainConfig) -> bool:
    return wandb_config.asdict(exclude=["wandb"]) == train_config.asdict(exclude=["wandb"])


def _get_wandb_config(wandb_run) -> TrainConfig:
    local_storage = LocalFileSystemAdapter()
    temp_file = local_storage.create_temp_file(suffix=".yaml")

    om.save(config=wandb_run.config, f=temp_file)
    wandb_config = TrainConfig.load(temp_file)

    return wandb_config


def _get_matching_wandb_runs(wandb_runs, training_run_dir: str) -> List:
    config_path = os.path.join(training_run_dir, CONFIG_YAML)
    local_config_path = cached_path(config_path)
    train_config = TrainConfig.load(local_config_path)

    return [
        wandb_run for wandb_run in wandb_runs if _are_equal_configs(_get_wandb_config(wandb_run), train_config)
    ]


def _get_wandb_path(run_dir: str) -> str:
    run_dir_storage = _get_storage_adapter_for_path(run_dir)

    config_path = os.path.join(run_dir, CONFIG_YAML)
    if not run_dir_storage.is_file(config_path):
        raise FileNotFoundError("No config file found in run dir, cannot get wandb path")

    local_config_path = cached_path(config_path)
    config = TrainConfig.load(local_config_path, validate_paths=False)

    if config.wandb is None or config.wandb.entity is None or config.wandb.project is None:
        raise ValueError(f"Run at {run_dir} has missing wandb config, cannot get wandb run path")

    wandb_runs = []

    wandb_dir = os.path.join(run_dir, "wandb/")
    if run_dir_storage.is_dir(wandb_dir):
        wandb_runs += _get_wandb_runs_from_wandb_dir(run_dir_storage, wandb_dir, config)

    wandb_runs += _get_wandb_runs_from_train_config(config)

    # Remove duplicate wandb runs based on run path, and wandb runs that do not match our run.
    wandb_runs = list({_get_wandb_path_from_run(wandb_run): wandb_run for wandb_run in wandb_runs}.values())
    wandb_matching_runs = _get_matching_wandb_runs(wandb_runs, run_dir)

    if len(wandb_matching_runs) == 0:
        raise RuntimeError(f"Failed to find any wandb runs for {run_dir}. Run might no longer exist")

    if len(wandb_matching_runs) > 1:
        wandb_run_urls = [wandb_run.url for wandb_run in wandb_matching_runs]
        raise RuntimeError(
            f"Found {len(wandb_matching_runs)} runs matching run dir {run_dir}, cannot determine correct run: {wandb_run_urls}"
        )

    return _get_wandb_path_from_run(wandb_matching_runs[0])


def _append_wandb_path(
    base_dir: str, run_dir_or_archive: str, append_archive_extension: bool = False, run_dir: Optional[str] = None
) -> str:
    run_dir_or_archive_storage = _get_storage_adapter_for_path(run_dir_or_archive)
    if run_dir is None:
        run_dir = _unarchive_if_archive(run_dir_or_archive, run_dir_or_archive_storage)

    wandb_path = _get_wandb_path(run_dir)

    if _is_archive(run_dir_or_archive, run_dir_or_archive_storage) and append_archive_extension:
        archive_extension = "".join(Path(run_dir_or_archive).suffixes)
        relative_wandb_path = wandb_path + archive_extension
    else:
        relative_wandb_path = wandb_path + "/"

    return os.path.join(base_dir, relative_wandb_path)


def _copy(src_path: str, dest_path: str, temp_dir: str):
    """
    Copies the entry at `src_path` to `dest_path`. The destination path can be a directory
    that does not exist (creating directories without a corresponding file is not always possible
    in cloud storage). In exchange we require that `src_path` and `dest_path` are either
    both files or both directories.
    """
    src_storage_type = StorageAdapter.get_storage_type_for_path(src_path)
    dest_storage_type = StorageAdapter.get_storage_type_for_path(dest_path)

    if src_storage_type == dest_storage_type and src_storage_type != StorageType.LOCAL_FS:
        # The current implementation downloads the src entry to local storage and then
        # uploads it to the destination. Downloading locally can likely be avoided when
        # moving an entry within the same storage.
        log.warning("Moving files and directories within the same storage system has not yet been optimized")

    src_storage = StorageAdapter.create_storage_adapter(src_storage_type)
    src_is_file = src_storage.is_file(src_path)
    src_is_dir = src_storage.is_dir(src_path)
    assert not (src_is_file and src_is_dir), f"Source {src_path} is both a file and a directory"

    dest_storage = StorageAdapter.create_storage_adapter(dest_storage_type)
    if dest_storage.is_file(dest_path):
        raise ValueError(f"A file already exists at destination {dest_path}")
    if src_is_file and (dest_path.endswith("/") or dest_storage.is_dir(dest_path)):
        raise ValueError(f"Source path {src_path} is a file but the destination {dest_path} is a directory.")

    local_path: PathOrStr
    if src_is_file:
        local_path = cached_path(src_path)
    elif src_is_dir:
        if src_storage_type == StorageType.LOCAL_FS:
            local_path = src_path
        else:
            local_storage = LocalFileSystemAdapter()
            local_path = local_storage.create_temp_dir(directory=temp_dir)
            log.info("Temporarily downloading %s to %s", src_path, local_path)
            src_storage.download_folder(src_path, local_path)
    else:
        raise ValueError(f"Source path {src_path} does not correspond to a valid file or directory")

    log.info("Uploading %s to %s", local_path, dest_path)
    dest_storage.upload(local_path, dest_path)


def _get_src_and_dest_for_copy(
    src_storage: StorageAdapter, run_dir_or_archive: str, dest_dir: str, config: MoveRunConfig
) -> Tuple[str, str]:
    is_archive_file = _is_archive(run_dir_or_archive, src_storage)
    # We need to unarchive the run if we want to get the wandb path
    should_unarchive = is_archive_file and (not config.store_archived or config.append_wandb_path)

    if is_archive_file and not should_unarchive:
        dest_file_path = os.path.join(dest_dir, Path(run_dir_or_archive).name)
        return run_dir_or_archive, dest_file_path

    run_dir = _unarchive_if_archive(run_dir_or_archive, src_storage)

    src_path = run_dir_or_archive if config.store_archived else run_dir

    dest_path: str
    if config.append_wandb_path:
        dest_path = _append_wandb_path(
            dest_dir, run_dir_or_archive, append_archive_extension=config.store_archived, run_dir=run_dir
        )
    elif is_archive_file and not config.store_archived:
        archive_extension = "".join(Path(run_dir_or_archive).suffixes)
        dir_name = Path(run_dir_or_archive).name.removesuffix(archive_extension)
        dest_path = os.path.join(dest_dir, dir_name)
    else:
        dest_path = dest_dir

    return src_path, dest_path


def _move_run(src_storage: StorageAdapter, run_dir_or_archive: str, dest_dir: str, config: MoveRunConfig):
    log.info("Moving run directory or archive %s to directory %s", run_dir_or_archive, dest_dir)

    dest_storage = _get_storage_adapter_for_path(dest_dir)
    if dest_storage.is_file(dest_dir):
        raise ValueError(f"Destination directory {dest_dir} is a file")

    src_move_path, dest_move_path = _get_src_and_dest_for_copy(src_storage, run_dir_or_archive, dest_dir, config)

    if src_move_path.rstrip("/") == dest_move_path.rstrip("/"):
        # This could be a valid scenario if the user is, for example, trying to
        # append wandb path to runs and this run has the right wandb path already.
        log.info("Source and destination move paths are both %s, skipping", src_move_path)
        return

    if config.dry_run:
        log.info("Would copy %s to %s", src_move_path, dest_move_path)
    else:
        log.info("Copying %s to %s", src_move_path, dest_move_path)
        _copy(src_move_path, dest_move_path, config.temp_dir)

    if not config.keep_src:
        if config.dry_run:
            log.info("Would delete run dir or archive %s", run_dir_or_archive)
        else:
            log.info("Deleting run dir or archive %s", run_dir_or_archive)
            src_storage.delete_path(run_dir_or_archive)


def move_run(run_path: str, dest_dir: str, config: MoveRunConfig):
    storage = _get_storage_adapter_for_path(run_path)
    run_dir_or_archive = _format_dir_or_archive_path(storage, run_path)
    dest_dir = f"{dest_dir}/" if not dest_dir.endswith("/") else dest_dir
    _move_run(storage, run_dir_or_archive, dest_dir, config)


def _add_cached_path_s3_client():
    class S3SchemeClient(S3Client):
        """
        A class that the `cached_path` module can use to retrieve resources from
        S3 (and R2, which is S3-based).  Refer to
        [cached_path docs](https://github.com/allenai/cached_path/blob/main/docs/source/overview.md#supported-url-schemes).
        """

        # This is used by cached_path to get the schemes are handled by this client
        scheme = ("s3", "r2")

        def __init__(self, resource: str) -> None:
            super().__init__(resource)
            parsed_path = urlparse(resource)
            bucket_name = parsed_path.netloc
            key = parsed_path.path.lstrip("/")

            profile_name = util._get_s3_profile_name(parsed_path.scheme)
            endpoint_url = util._get_s3_endpoint_url(parsed_path.scheme)

            session = boto3.session.Session(profile_name=profile_name)
            s3_resource = session.resource("s3", endpoint_url=endpoint_url)
            self.s3_object = s3_resource.Object(bucket_name, key)  # type: ignore

    add_scheme_client(S3SchemeClient)


def _setup_cached_path(temp_dir: str):
    if temp_dir is not None:
        set_cache_dir(temp_dir)

    _add_cached_path_s3_client()


def perform_operation(args: argparse.Namespace):
    if args.dry_run:
        log.info("Dry run, no irreversible actions will be taken")

    if (
        args.temp_dir is not None
        and StorageAdapter.get_storage_type_for_path(args.temp_dir) != StorageType.LOCAL_FS
    ):
        raise ValueError("Temporary directory must be a local path")

    temp_dir = tempfile.mkdtemp(dir=args.temp_dir)
    _setup_cached_path(temp_dir)

    try:
        if args.op == CleaningOperations.DELETE_BAD_RUNS:
            delete_bad_runs_config = DeleteBadRunsConfig(
                dry_run=args.dry_run,
                temp_dir=temp_dir,
                should_check_is_run=args.should_check_is_run,
                ignore_non_runs=args.ignore_non_runs,
                max_archive_size=int(args.max_archive_size),
            )
            if args.run_paths is not None:
                delete_bad_runs(args.run_paths, delete_bad_runs_config)
            else:
                raise ValueError("Run paths not provided for run cleaning")
        elif args.op == CleaningOperations.UNSHARD_CHECKPOINTS:
            unshard_checkpoints_config = UnshardCheckpointsConfig(
                dry_run=args.dry_run,
                temp_dir=temp_dir,
                latest_checkpoint_only=args.latest_checkpoint_only,
                delete_sharded_checkpoints=args.delete_sharded_checkpoints,
                checkpoint_num=args.checkpoint_num,
            )
            if args.run_path is not None:
                unshard_run_checkpoints(args.run_path, args.dest_dir, unshard_checkpoints_config)
            else:
                raise ValueError("Run path not provided for unsharding")
        elif args.op == CleaningOperations.MOVE_RUN:
            move_run_config = MoveRunConfig(
                dry_run=args.dry_run,
                temp_dir=temp_dir,
                append_wandb_path=args.append_wandb_path,
                keep_src=args.keep_src,
                store_archived=args.store_archived,
            )
            if args.run_path is not None and args.dest_dir is not None:
                move_run(args.run_path, args.dest_dir, move_run_config)
            else:
                raise ValueError("Run path or dest dir not provided for moving run")
        else:
            raise NotImplementedError(args.op)
    finally:
        if Path(temp_dir).is_dir():
            log.info("Deleting temp dir %s", temp_dir)
            shutil.rmtree(temp_dir)


def _add_delete_subparser(subparsers: _SubParsersAction):
    delete_runs_parser: ArgumentParser = subparsers.add_parser(
        "clean", help="Delete bad runs (e.g. runs with no non-trivial checkpoints)"
    )
    delete_runs_parser.set_defaults(op=CleaningOperations.DELETE_BAD_RUNS)

    delete_runs_parser.add_argument(
        "run_paths",
        nargs="+",
        help="Directory or archive file paths corresponding to runs.",
    )

    run_verification_parser = delete_runs_parser.add_mutually_exclusive_group(required=False)
    run_verification_parser.add_argument(
        "--bypass_run_verification",
        action="store_false",
        dest="should_check_is_run",
        help="(UNSAFE) Bypass the sanity check that a directory/archive being deleted is a run. This could result in a non-run directory/archive being deleted.",
    )
    run_verification_parser.add_argument(
        "--ignore_non_runs",
        action="store_true",
        help="Ignore (do not delete) directories/archives that are not runs.",
    )

    delete_runs_parser.add_argument(
        "--max_archive_size",
        type=float,
        default=DEFAULT_DELETE_MAX_ARCHIVE_SIZE,
        help="Max size archive files to consider for deletion (in bytes). Any archive larger than this is ignored/not deleted.",
    )


def _add_unsharding_subparser(subparsers: _SubParsersAction):
    unsharding_runs_parser: ArgumentParser = subparsers.add_parser("unshard", help="unshard checkpoints of a run")
    unsharding_runs_parser.set_defaults(op=CleaningOperations.UNSHARD_CHECKPOINTS)

    unsharding_runs_parser.add_argument(
        "run_path",
        help="Path to run directory or archive containing checkpoints to unshard.",
    )
    unsharding_runs_parser.add_argument(
        "dest_dir",
        help="Path to directory where the run's unsharded checkpoints should be output (only the unsharded checkpoints are stored).",
    )
    unsharding_runs_parser.add_argument(
        "--latest_checkpoint_only",
        action="store_true",
        help="If set, only the latest checkpoint of each run (if sharded) is unsharded.",
    )
    unsharding_runs_parser.add_argument(
        "--delete_sharded",
        dest="delete_sharded_checkpoints",
        action="store_true",
        help="If set, deletes sharded checkpoints after they have been successfully unsharded.",
    )
    unsharding_runs_parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=None,
        help="If provided, unsharding is restricted to this checkpoint of the run.",
    )


def _add_move_subparser(subparsers: _SubParsersAction):
    move_parser: ArgumentParser = subparsers.add_parser("move", help="move run to a new location")
    move_parser.set_defaults(op=CleaningOperations.MOVE_RUN)

    move_parser.add_argument(
        "run_path",
        help="Path of run directory or archive to move.",
    )
    move_parser.add_argument(
        "dest_dir",
        help="Path of directory to which the run should be moved.",
    )
    move_parser.add_argument(
        "--keep_src",
        action="store_true",
        help="If set, the run is not removed from the source location.",
    )
    move_parser.add_argument(
        "--unarchive",
        dest="store_archived",
        action="store_false",
        help="If set and the run path corresponds to an archive file, then the unarchived form of the run is stored at the destination.",
    )
    move_parser.add_argument(
        "--append_wandb_path",
        action="store_true",
        help="If set, the wandb path for the run is found and appended to the destination dir. If the run is being stored as an archive file, wandb id is first removed from the wandb path and used as the filename.",
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
        "--temp_dir",
        help="Local directory where artifacts (e.g. unarchived directories) can be stored temporarily",
    )

    subparsers = parser.add_subparsers(dest="command", help="Cleaning commands", required=True)
    _add_delete_subparser(subparsers)
    _add_unsharding_subparser(subparsers)
    _add_move_subparser(subparsers)

    return parser


def main():
    args = get_parser().parse_args()

    util.prepare_cli_environment()
    perform_operation(args)


if __name__ == "__main__":
    main()
