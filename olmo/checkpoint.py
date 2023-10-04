import io
import logging
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed.checkpoint as dist_cp
from packaging import version
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import WriteResult, _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.futures import Future

from ._checkpoint_optimizer import load_sharded_optimizer_state_dict
from .aliases import PathOrStr
from .optim import Optimizer, fix_optim_state_dict
from .util import (
    barrier,
    dir_is_empty,
    get_bytes_range,
    get_fs_local_rank,
    resource_path,
    upload,
)

__all__ = [
    "save_fsdp_model_and_optim_state",
    "load_fsdp_model_and_optim_state",
    "save_state_dict",
    "load_state_dict",
    "load_model_state",
    "RemoteFileSystemWriter",
    "RemoteFileSystemReader",
]


log = logging.getLogger(__name__)

MODEL_AND_OPTIM_FOLDER = "model_and_optim"


def save_fsdp_model_and_optim_state(
    checkpoint_dir: PathOrStr,
    fsdp_model: FSDP,
    optim: Optimizer,
    *,
    upload_to: Optional[str] = None,
    save_overwrite: bool = False,
):
    """
    Use this to save a state dict for an FSDP model and its optimizer via :module:`torch.distributed.checkpoint`
    functions. This should be used during distributed training and should be called by all ranks.

    :param checkpoint_dir: The directory to save to.
    :param fsdp_model: The FSDP model.
    :param optim: The FSDP model's optimizer.
    :param upload_to: Optional, a remote "directory" to upload the checkpoint files to.
    :param save_overwrite: Overwrite existing files.

    :raises FileExistsError: If a model and optim checkpoint already exists in ``checkpoint_dir`` and ``save_overwrite=False``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    target_dir = checkpoint_dir / MODEL_AND_OPTIM_FOLDER
    if save_overwrite:
        if get_fs_local_rank() == 0:
            shutil.rmtree(target_dir, ignore_errors=True)
    elif not dir_is_empty(target_dir):
        raise FileExistsError(target_dir)
    barrier()
    if get_fs_local_rank() == 0:
        target_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    with FSDP.state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        model_and_optim_state = {
            "model": fsdp_model.state_dict(),
            "optim": FSDP.optim_state_dict(fsdp_model, optim),
        }
        dist_cp.save_state_dict(
            model_and_optim_state,
            RemoteFileSystemWriter(
                target_dir,
                upload_to=None if upload_to is None else f"{upload_to.rstrip('/')}/{MODEL_AND_OPTIM_FOLDER}",
                save_overwrite=save_overwrite,
            ),
        )


def load_fsdp_model_and_optim_state(
    checkpoint_dir: PathOrStr, fsdp_model: FSDP, optim: Optimizer, *, local_cache: Optional[PathOrStr] = None
):
    """
    Use this to load a state dict for an FSDP model and its optimizer via :module:`torch.distributed.checkpoint`
    functions. This should be used during distributed training and should be called by all ranks.

    :param checkpoint_dir: The checkpoint directory to load from. This can be a local or remote directory.
    :param fsdp_model: The FSDP model.
    :param optim: The FSDP model's optimizer.
    :param local_cache: A local cache of the checkpoint directory. Use this when the ``checkpoint_dir`` is a
        remote "directory" but there might be a cached version of the same artifacts.

    :raises FileNotFoundError: If the ``checkpoint_dir`` doesn't contain a model and optimizer checkpoint.
    """
    load_path = str(checkpoint_dir).rstrip("/")
    local_cache = None if local_cache is None else Path(local_cache)
    with FSDP.state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        # Load the model state dict in place.
        log.info("Loading model state...")
        model_state = {"model": fsdp_model.state_dict()}
        dist_cp.load_state_dict(
            model_state,
            RemoteFileSystemReader(
                f"{load_path}/{MODEL_AND_OPTIM_FOLDER}",
                local_cache=None if local_cache is None else local_cache / MODEL_AND_OPTIM_FOLDER,
            ),
        )
        fsdp_model.load_state_dict(model_state["model"])

        # Load optim state dict in place.
        log.info("Loading optimizer state...")
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=model_state["model"],
            optimizer_key="optim",
            storage_reader=RemoteFileSystemReader(
                f"{load_path}/{MODEL_AND_OPTIM_FOLDER}",
                local_cache=None if local_cache is None else local_cache / MODEL_AND_OPTIM_FOLDER,
            ),
        )
        # NOTE: Careful! The order of the these arguments has changed from 2.0 to 2.1... ¯\_(ツ)_/¯
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], fsdp_model, optim)  # type: ignore
        else:
            flattened_osd = FSDP.optim_state_dict_to_load(fsdp_model, optim, optim_state["optim"])  # type: ignore
        optim.load_state_dict(fix_optim_state_dict(optim, flattened_osd))


def save_state_dict(
    checkpoint_dir: PathOrStr,
    fname: str,
    state_dict: Dict[str, Any],
    *,
    upload_to: Optional[str] = None,
    save_overwrite: bool = False,
):
    """
    Save a regular state dict to the file ``fname`` within ``checkpoint_dir`` using :func:`torch.save()`.
    This can be used during distributed training or not. If during distributed training the ``fname`` should be unique
    for each rank.

    :param checkpoint_dir: The directory to save to.
    :param fname: The target file within ``checkpoint_dir`` to save to. This should be a path relative to the ``checkpoint_dir``.
    :param state_dict: The state dict to save.
    :param upload_to: Optional, a remote "directory" to upload the file to.
    :param save_overwrite: Overwrite existing files.

    :raises FileExistsError: If the ``fname`` already exists within ``checkpoint_dir`` and ``save_overwrite=False``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    target_path = checkpoint_dir / fname
    if save_overwrite:
        target_path.unlink(missing_ok=True)
    elif target_path.is_file():
        raise FileExistsError(target_path)
    barrier()
    target_path.parent.mkdir(exist_ok=True, parents=True)
    barrier()
    torch.save(state_dict, target_path)
    if upload_to is not None:
        upload_target = f"{upload_to.rstrip('/')}/{fname}"
        log.info(f"Uploading {target_path} to {upload_target}...")
        upload(target_path, upload_target, save_overwrite=save_overwrite)


def load_state_dict(checkpoint_dir: PathOrStr, fname: str, *, local_cache: Optional[PathOrStr] = None):
    """
    Load a regular state dict from the file ``fname`` within ``checkpoint_dir`` using :func:`torch.load()`.
    This can be used during distributed training or not.

    :param checkpoint_dir: A local or remote checkpoint directory.
    :param fname: The target file within the ``checkpoint_dir``. This should be a path relative to the ``checkpoint_dir``.
    :param local_cache: A local cache of the checkpoint directory. Use this when the ``checkpoint_dir`` is a
        remote "directory" but there might be a cached version of the same artifacts.

    :raises FileNotFoundError: If ``fname`` doesn't exist in the ``checkpoint_dir`` or the local cache.
    """
    return torch.load(resource_path(str(checkpoint_dir).rstrip("/"), fname, local_cache=local_cache))


def load_model_state(checkpoint_dir: PathOrStr, model: torch.nn.Module):
    """
    Load model state from a distributed FSDP model checkpoint created from :func:`save_fsdp_model_and_optim_state()`.
    Note that ``model`` should not be wrapped with FSDP.
    """
    state_dict = {"model": model.state_dict()}
    dist_cp.load_state_dict(
        state_dict,
        RemoteFileSystemReader(f"{str(checkpoint_dir).rstrip('/')}/{MODEL_AND_OPTIM_FOLDER}"),
        no_dist=True,
    )
    model.load_state_dict(state_dict["model"])


def _default_thread_count() -> int:
    return min(8, (os.cpu_count() or 1) + 4)


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
        thread_count: Optional[int] = None,
        per_thread_copy_ahead: int = 10_000_000,
        upload_to: Optional[str] = None,
        save_overwrite: bool = False,
    ) -> None:
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread count must be at least 1")
        super().__init__(
            path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count or _default_thread_count(),
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

    def __init__(
        self, path: PathOrStr, *, local_cache: Optional[PathOrStr] = None, thread_count: Optional[int] = None
    ):
        super().__init__()
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread count must be at least 1")
        self.path = str(path).rstrip("/")
        self.cache = None if local_cache is None else Path(local_cache)
        self.thread_count = thread_count or _default_thread_count()
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self._metadata: Optional[Metadata] = None

    def _get_bytes(self, relative_path: str, offset: int, length: int) -> bytes:
        if self.cache is not None and (path := self.cache / relative_path).is_file():
            return get_bytes_range(path, offset, length)
        else:
            return get_bytes_range(f"{self.path}/{relative_path}", offset, length)

    def _get_content_for_read(self, read_item: ReadItem) -> Tuple[ReadItem, bytes]:
        sinfo = self.storage_data[read_item.storage_index]
        content = self._get_bytes(sinfo.relative_path, sinfo.offset, sinfo.length)
        return (read_item, content)

    def read_data(self, plan: dist_cp.LoadPlan, planner: dist_cp.LoadPlanner) -> Future[None]:
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            read_item_content_futures = []
            for read_item in plan.items:
                read_item_content_futures.append(executor.submit(self._get_content_for_read, read_item))
            read_item_content_results = []
            for f in as_completed(read_item_content_futures):
                read_item_content_results.append(f.result())

        # Modified from `FileSystemReader.read_data()`
        for read_item, content in read_item_content_results:
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
        if self._metadata is None:
            with resource_path(self.path, ".metadata", local_cache=self.cache).open("rb") as metadata_file:
                self._metadata = pickle.load(metadata_file)
        return self._metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        del is_coordinator
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: dist_cp.LoadPlan) -> dist_cp.LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[dist_cp.LoadPlan]) -> List[dist_cp.LoadPlan]:
        return global_plan
