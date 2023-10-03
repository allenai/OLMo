"""
Custom distributed checkpointing.
"""

import io
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, cast

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import WriteResult, _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType
from torch.futures import Future

from .aliases import PathOrStr
from .util import get_bytes_range, resource_path, upload

__all__ = ["RemoteFileSystemWriter", "RemoteFileSystemReader"]


log = logging.getLogger(__name__)


class RemoteFileSystemWriter(dist_cp.FileSystemWriter):
    """
    A subclass of :class:`~torch.distributed.checkpoint.FileSystemWriter` that can upload files
    directly to a cloud bucket when ``upload_to`` is specified.
    """

    def __init__(
        self,
        path: PathOrStr,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        upload_to: Optional[str] = None,
        save_overwrite: bool = False,
    ) -> None:
        super().__init__(
            path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
        )
        self.upload_to = None if upload_to is None else upload_to.rstrip("/")
        self.save_overwrite = save_overwrite

    def write_data(
        self,
        plan: dist_cp.SavePlan,
        planner: dist_cp.SavePlanner,
    ) -> Future[List[WriteResult]]:
        fut = super().write_data(plan, planner)
        if self.upload_to is not None:
            files_to_upload = set()
            for write_result in fut.wait():
                files_to_upload.add(write_result.storage_data.relative_path)

            with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                futures = []
                for fname in files_to_upload:
                    source = self.path / fname
                    target = f"{self.upload_to}/{fname}"
                    log.info(f"Uploading {source} to {target}...")
                    futures.append(executor.submit(upload, source, target, save_overwrite=self.save_overwrite))
                for f in as_completed(futures):
                    f.result()
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        super().finish(metadata, results)
        if self.upload_to is not None:
            source = self.path / ".metadata"
            target = f"{self.upload_to}/.metadata"
            log.info(f"Uploading {source} to {target}...")
            upload(source, target, save_overwrite=self.save_overwrite)


class RemoteFileSystemReader(dist_cp.StorageReader):
    """
    A :class:`~torch.distributed.checkpoint.StorageReader` based on :class:`~torch.distributed.checkpoint.FileSystemReader`
    that can read data directly from cloud storage as well as a local directory.
    """

    def __init__(self, path: PathOrStr):
        super().__init__()
        self.path = str(path).rstrip("/")
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()

    def read_data(self, plan: dist_cp.LoadPlan, planner: dist_cp.LoadPlanner) -> Future[None]:
        # Modified from `FileSystemReader.read_data()`
        for read_item in plan.items:
            sinfo = self.storage_data[read_item.storage_index]
            content = get_bytes_range(f"{self.path}/{sinfo.relative_path}", sinfo.offset, sinfo.length)
            bytes = io.BytesIO(content)
            bytes.seek(0)
            if read_item.type == LoadItemType.BYTE_IO:
                planner.load_bytes(read_item, bytes)
            else:
                tensor = cast(torch.Tensor, torch.load(bytes, map_location="cpu"))
                tensor = narrow_tensor_by_index(tensor, read_item.storage_offsets, read_item.lengths)
                target_tensor = planner.resolve_tensor(read_item).detach()

                assert (
                    target_tensor.size() == tensor.size()
                ), f"req {read_item.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        with resource_path(self.path, ".metadata").open("rb") as metadata_file:
            return pickle.load(metadata_file)

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        del is_coordinator
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: dist_cp.LoadPlan) -> dist_cp.LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[dist_cp.LoadPlan]) -> List[dist_cp.LoadPlan]:
        return global_plan
