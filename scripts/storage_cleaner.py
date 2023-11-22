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
import botocore.client
import botocore.exceptions as boto_exceptions
import google.cloud.storage as gcs
from cached_path import add_scheme_client, cached_path
from cached_path.schemes import S3Client
from google.api_core.exceptions import NotFound

from olmo import util
from olmo.aliases import PathOrStr

log = logging.getLogger(__name__)


DEFAULT_MAX_ARCHIVE_SIZE: float = 5 * 1024 * 1024 * 1024  # 5GB


class CleaningOperations(Enum):
    DELETE_BAD_RUNS = auto()


class StorageType(Enum):
    LOCAL_FS = auto()
    GCS = auto()
    S3 = auto()
    R2 = auto()


class StorageAdapter(ABC):
    @abstractmethod
    def list_entries(self, path: str, full_path: bool = False, max_file_size: Optional[int] = None) -> List[str]:
        """Lists all the entries within the directory or archive file at the given path.
        Returns only top-level entries (i.e. not entries in subdirectories).

        `full_path`: If `full_path` is set to true, returned entries are valid full paths. These full paths
        might not have `path` as their parent (for example, if `path` corresponds to an archive file).
        If `full_path` is set to false, returned entries are file/directory names.

        `max_file_size`: Sets a threshold (in bytes) for the largest size file to retain within entries.
        Any file of larger size is not included in the returned results.
        """

    @abstractmethod
    def list_dirs(self, path: str, full_path: bool = False) -> List[str]:
        """Lists all the directories within the directory or archive file at the given path.
        Returns only top-level entries (i.e. not entries in subdirectories).

        `full_path`: See `list_entries` for details.
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

    @classmethod
    def create_storage_adapter(cls, storage_type: StorageType):
        if storage_type == StorageType.LOCAL_FS:
            return LocalFileSystemAdapter()
        if storage_type == StorageType.GCS:
            return GoogleCloudStorageAdapter()
        if storage_type == StorageType.S3:
            return S3StorageAdapter(storage_type)
        if storage_type == StorageType.R2:
            r2_account_id = os.environ.get("R2_ACCOUNT_ID")
            if r2_account_id is None:
                raise ValueError(
                    "R2_ACCOUNT_ID environment variable not set with R2 account id, cannot connect to R2"
                )
            return S3StorageAdapter(storage_type, endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com")

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
        self._archive_extensions: List[str] = [
            extension.lower() for _, extensions, _ in shutil.get_unpack_formats() for extension in extensions
        ]

    def __del__(self):
        for temp_file in self._temp_files:
            temp_file.close()

    def create_temp_file(self, suffix: Optional[str] = None) -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
        self._temp_files.append(temp_file)
        return temp_file.name

    def has_supported_archive_extension(self, path: PathOrStr) -> bool:
        filename = Path(path).name.lower()
        return any(filename.endswith(extension) for extension in self._archive_extensions)

    def _list_entries(
        self, path: PathOrStr, full_path: bool, include_files: bool = True, max_file_size: Optional[int] = None
    ) -> List[str]:
        path = Path(path)
        if path.is_file():
            if not self.has_supported_archive_extension(path):
                raise ValueError(f"File does not have a supported archive extension: {path}")

            path = cached_path(path, extract_archive=True)

        if path.is_dir():
            return [
                str(entry) if full_path else entry.name
                for entry in path.iterdir()
                if (
                    (include_files or not entry.is_file())
                    and (
                        not entry.is_file() or max_file_size is None or self._get_file_size(entry) <= max_file_size
                    )
                )
            ]

        raise ValueError(f"Path does not correspond to directory or supported archive file: {path}")

    def list_entries(self, path: str, full_path: bool = False, max_file_size: Optional[int] = None) -> List[str]:
        return self._list_entries(path, full_path=full_path, max_file_size=max_file_size)

    def list_dirs(self, path: str, full_path: bool = False) -> List[str]:
        return self._list_entries(path, full_path=full_path, include_files=False)

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
        full_path: bool,
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

        if full_path:
            entries = [self._get_path(bucket_name, entry) for entry in entries]
        else:
            entries = [entry.removeprefix(key) for entry in entries]

        return entries


    def _list_entries(
        self, path: str, full_path: bool, include_files: bool = True, max_file_size: Optional[int] = None
    ) -> List[str]:
        bucket_name, key = self._get_bucket_name_and_key(path)

        if self.local_fs_adapter.has_supported_archive_extension(path):
            file_path = str(cached_path(path, extract_archive=True))

            if not include_files:
                return self.local_fs_adapter.list_dirs(file_path, full_path=full_path)
            return self.local_fs_adapter.list_entries(file_path, full_path=full_path, max_file_size=max_file_size)

        if self._is_file(bucket_name, key):
            raise ValueError(f"Path corresponds to a file without a supported archive extension {path}")

        res = self._get_directory_entries(
            bucket_name, key, full_path, include_files=include_files, max_file_size=max_file_size
        )
        return res

    def list_entries(self, path: str, full_path: bool = False, max_file_size: Optional[int] = None) -> List[str]:
        return self._list_entries(path, full_path=full_path, max_file_size=max_file_size)

    def list_dirs(self, path: str, full_path: bool = False) -> List[str]:
        return self._list_entries(path, full_path=full_path, include_files=False)

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

    def is_dir(self, path: str) -> bool:
        path = f"{path}/" if not path.endswith("/") else path
        bucket_name, key = self._get_bucket_name_and_key(path)
        bucket = self.gcs_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=key, max_results=1))

        return not self._is_file(bucket_name, key) and len(blobs) > 0


class S3StorageAdapter(StorageAdapter):
    def __init__(self, storage_type: StorageType, endpoint_url: Optional[str] = None):
        super().__init__()
        self._storage_type = storage_type
        self._s3_client = util._get_s3_client(endpoint_url=endpoint_url)

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
        full_path: bool,
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

        if full_path:
            entries = [self._get_path(bucket_name, entry) for entry in entries]
        else:
            entries = [entry.removeprefix(key) for entry in entries]

        return entries

    def _list_entries(
        self, path: str, full_path: bool, include_files: bool = True, max_file_size: Optional[int] = None
    ) -> List[str]:
        bucket_name, key = self._get_bucket_name_and_key(path)

        if self.local_fs_adapter.has_supported_archive_extension(path):
            file_path = str(cached_path(path, extract_archive=True))

            if not include_files:
                return self.local_fs_adapter.list_dirs(file_path, full_path=full_path)
            return self.local_fs_adapter.list_entries(file_path, full_path=full_path, max_file_size=max_file_size)

        if self._is_file(bucket_name, key):
            raise ValueError(f"Path corresponds to a file without a supported archive extension {path}")

        res = self._get_directory_entries(
            bucket_name, key, full_path, include_files=include_files, max_file_size=max_file_size
        )
        return res

    def list_entries(self, path: str, full_path: bool = False, max_file_size: Optional[int] = None) -> List[str]:
        return self._list_entries(path, full_path=full_path, max_file_size=max_file_size)

    def list_dirs(self, path: str, full_path: bool = False) -> List[str]:
        return self._list_entries(path, full_path=full_path, include_files=False)

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
        if self._is_file(bucket_name, key):
            return False

        response = self._s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key, MaxKeys=1)
        return "Contents" in response

    def is_dir(self, path: str) -> bool:
        path = f"{path}/" if not path.endswith("/") else path
        bucket_name, key = self._get_bucket_name_and_key(path)

        return self._is_dir(bucket_name, key)


@dataclass
class DeleteBadRunsConfig:
    dry_run: bool
    should_check_is_run: bool
    ignore_non_runs: bool
    max_archive_size: Optional[int]


def _get_storage_adapter_for_path(path: str) -> StorageAdapter:
    storage_type = StorageAdapter.get_storage_type_for_path(path)
    return StorageAdapter.create_storage_adapter(storage_type)


def _contains_checkpoint_dir(dir_entries: List[str]) -> bool:
    return any(re.match(r"step\d+(-unsharded)?", entry) is not None for entry in dir_entries)


def _contains_nontrivial_checkpoint_dir(dir_entries: List[str]) -> bool:
    return any(re.match(r"step[1-9]\d*(-unsharded)?", entry) is not None for entry in dir_entries)


def _is_run(run_path: str, run_entries: Optional[List[str]] = None) -> bool:
    """
    This method is best effort. It may mark run paths as not (false negatives) or mark non-run
    paths as runs (false positives). We prioritize minimizing false positives.
    """
    if run_entries is None:
        storage = _get_storage_adapter_for_path(run_path)
        run_entries = storage.list_entries(run_path)

    return _contains_checkpoint_dir(run_entries)


def _verify_non_run_deletion(run_dir_or_archive: str, run_entries: List[str], config: DeleteBadRunsConfig):
    msg = f"Attempting to delete non-run directory or archive {run_dir_or_archive} (first 5 entries: {run_entries[:5]})."
    if config.ignore_non_runs:
        log.warning(msg)
        return

    raise ValueError(msg)


def _format_dir_or_archive_path(storage: StorageAdapter, path: str) -> str:
    if storage.is_file(path):
        local_fs_adapter = LocalFileSystemAdapter()
        if not local_fs_adapter.has_supported_archive_extension(path):
            raise ValueError(f"Path corresponds to a non-archive file: {path}")

        return path

    if storage.is_dir(path):
        return f"{path}/" if not path.endswith("/") else path

    raise ValueError(f"Path does not correspond to a directory or file: {path}")


def _get_archive_run_entry_paths(run_archive_path: str, storage: StorageAdapter, full_path: bool = False, max_file_size: Optional[int] = None) -> List[str]:
    # The unarchived file could have a redundant top-level directory. If the top-level
    # directory has only a directory, we should return that directory's entries instead.
    # We do not pass max_file_size to avoid accidentally skipping files.
    entry_paths = storage.list_entries(run_archive_path, full_path=True)
    if len(entry_paths) == 1:
        entry_path = entry_paths[0]
        assert StorageAdapter.get_storage_type_for_path(entry_path) == StorageType.LOCAL_FS, "Entries of archived files are expected to be local"
        local_storage = LocalFileSystemAdapter()
        if local_storage.is_dir(entry_path):
            return local_storage.list_entries(entry_path, full_path=full_path, max_file_size=max_file_size)

    return storage.list_entries(run_archive_path, full_path=full_path, max_file_size=max_file_size)


def _get_run_entries(run_dir_or_archive: str, storage: StorageAdapter, full_path: bool = False, max_file_size: Optional[int] = None) -> List[str]:
    local_storage = LocalFileSystemAdapter()
    if local_storage.has_supported_archive_extension(run_dir_or_archive):
        return _get_archive_run_entry_paths(run_dir_or_archive, storage, full_path=full_path, max_file_size=max_file_size)

    return storage.list_entries(run_dir_or_archive, full_path=full_path, max_file_size=max_file_size)


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

    run_entries = _get_run_entries(run_dir_or_archive, storage)

    if config.should_check_is_run and not _is_run(run_dir_or_archive, run_entries=run_entries):
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


def perform_operation(args: argparse.Namespace):
    if args.dry_run:
        log.info("Dry run, no irreversible actions will be taken")

    if args.op == CleaningOperations.DELETE_BAD_RUNS:
        delete_bad_runs_config = DeleteBadRunsConfig(
            dry_run=args.dry_run,
            should_check_is_run=args.should_check_is_run,
            ignore_non_runs=args.ignore_non_runs,
            max_archive_size=args.max_archive_size,
        )
        if args.run_paths is not None:
            delete_bad_runs(args.run_paths, delete_bad_runs_config)
        else:
            raise ValueError("Run paths not provided for run cleaning")
    else:
        raise NotImplementedError(args.op)


def _add_cached_path_r2_client(r2_account_id: str):
    endpoint_url = f"https://{r2_account_id}.r2.cloudflarestorage.com"

    class R2SchemeClient(S3Client):
        """
        A class that the `cached_path` module can use to retrieve resources from
        R2. Refer to
        [cached_path docs](https://github.com/allenai/cached_path/blob/main/docs/source/overview.md#supported-url-schemes).
        """

        scheme = "r2"

        def __init__(self, resource: str) -> None:
            super().__init__(resource)
            parsed_path = urlparse(resource)
            bucket_name = parsed_path.netloc
            r2_path = parsed_path.path.lstrip("/")

            session = boto3.session.Session()
            if session.get_credentials() is None:
                # Use unsigned requests.
                s3_resource = session.resource(
                    "s3",
                    endpoint_url=endpoint_url,
                    config=botocore.client.Config(signature_version=botocore.UNSIGNED),
                )
            else:
                s3_resource = session.resource("s3", endpoint_url=endpoint_url)
            self.s3_object = s3_resource.Object(bucket_name, r2_path)  # type: ignore

    add_scheme_client(R2SchemeClient)


def _add_cached_path_scheme_clients():
    r2_account_id = os.environ.get("R2_ACCOUNT_ID")
    if r2_account_id is not None:
        _add_cached_path_r2_client(r2_account_id)


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
        default=DEFAULT_MAX_ARCHIVE_SIZE,
        help="Max size archive files to consider for deletion (in bytes). Any archive larger than this is ignored/not deleted.",
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--dry_run",
        action="store_true",
        help="If set, indicate actions but do not do them",
    )

    subparsers = parser.add_subparsers(dest="command", help="Cleaning commands", required=True)
    _add_delete_subparser(subparsers)

    return parser


def main():
    args = get_parser().parse_args()

    util.prepare_cli_environment()
    _add_cached_path_scheme_clients()
    perform_operation(args)


if __name__ == "__main__":
    main()
