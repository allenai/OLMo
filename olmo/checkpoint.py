import gc
import io
import logging
import pickle
import shutil
import traceback
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import reduce
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, cast

import numpy as np
import torch
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp
import torch.nn as nn
from packaging import version
from torch.distributed import _remote_device
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.checkpoint.filesystem import WriteResult, _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.futures import Future
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp.flat_param import FlatParamHandle  # type: ignore
except ModuleNotFoundError:
    from torch.distributed.fsdp._flat_param import FlatParamHandle  # type: ignore

from olmo import util

from .aliases import PathOrStr
from .config import BaseConfig, ShardedCheckpointerType, TrainConfig
from .exceptions import OLMoCheckpointError
from .optim import Optimizer, fix_optim_state_dict
from .safetensors_util import safetensors_file_to_state_dict
from .torch_util import (
    SingleAccelerator,
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
)
from .util import (
    _get_s3_client,
    default_thread_count,
    dir_is_empty,
    get_bytes_range,
    get_progress_bar,
    resource_path,
    upload,
    wait_for,
)

__all__ = [
    "save_fsdp_model_and_optim_state",
    "load_fsdp_model_and_optim_state",
    "load_fsdp_optim_state",
    "save_state_dict",
    "load_state_dict",
    "load_model_state",
    "RemoteFileSystemWriter",
    "RemoteFileSystemReader",
    "Checkpointer",
    "FullCheckpointer",
    "TorchNewStyleShardedCheckpointer",
    "TorchLegacyShardedCheckpointer",
    "LocalShardedCheckpointer",
    "build_sharded_checkpointer",
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
    checkpoint_dir: PathOrStr,
    fsdp_model: FSDP,
    optim: Optimizer,
    *,
    local_cache: Optional[PathOrStr] = None,
    load_optimizer_state: bool = True,
):
    """
    Use this to load a state dict for an FSDP model and its optimizer via :module:`torch.distributed.checkpoint`
    functions. This should be used during distributed training and should be called by all ranks.

    :param checkpoint_dir: The checkpoint directory to load from. This can be a local or remote directory.
    :param fsdp_model: The FSDP model.
    :param optim: The FSDP model's optimizer.
    :param local_cache: A local cache of the checkpoint directory. Use this when the ``checkpoint_dir`` is a
        remote "directory" but there might be a cached version of the same artifacts.
    :param load_optimizer_state: Set to ``False`` to skip loading the optimizer state.

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

        if not load_optimizer_state:
            return

        # Load optim state dict in place.
        log.info("Loading sharded optimizer state...")
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=model_state["model"],
            optimizer_key="optim",
            storage_reader=RemoteFileSystemReader(
                f"{load_path}/{MODEL_AND_OPTIM_FOLDER}",
                local_cache=None if local_cache is None else local_cache / MODEL_AND_OPTIM_FOLDER,
            ),
        )
        # optim_state["optim"] = {
        #    'state': { fqn: { 'grad_norm_exp_avg': Tensor, 'step': Tensor, 'exp_avg': ShardedTensor, 'exp_avg_sq': ShardedTensor } },
        #    'param_groups': [{ 'param_names': [ fsdp_fqn, ... ], 'params': [ fqn, ... ], ... }],
        # }
        del model_state

        # Make sure tensors are on CPU! PyTorch puts them on GPU even though we have `offload_to_cpu=True`.
        for state in optim_state["optim"]["state"].values():
            for k in state.keys():
                state[k] = state[k].cpu()
        gc_cuda()

        load_fsdp_optim_state(fsdp_model, optim, optim_state["optim"])


def load_fsdp_optim_state(fsdp_model: FSDP, optim: Optimizer, optim_state: Dict[str, Any]):
    log.info("Flattening sharded optimizer state...")
    # flattened_osd = {
    #    'state': { id: { 'grad_norm_exp_avg': Tensor, 'step': Tensor, 'exp_avg': Tensor, 'exp_avg_sq': Tensor } },
    #    'param_groups': [{ 'param_names': [ fsdp_fqn, ... ], 'params': [ id, ... ], ... }],
    # }
    # NOTE: Careful! The order of the these arguments has changed from 2.0 to 2.1... ¯\_(ツ)_/¯
    if version.parse(torch.__version__) < version.parse("2.1.0"):
        flattened_osd = FSDP.optim_state_dict_to_load(optim_state, fsdp_model, optim)  # type: ignore
    else:
        flattened_osd = FSDP.optim_state_dict_to_load(fsdp_model, optim, optim_state)  # type: ignore

    del optim_state
    gc_cuda()

    log.info("Loading flattened optimizer state...")

    # Put optim state on CPU since `Optimizer.load_state_dict()` will create a deepcopy of the whole state dict,
    # which takes up unnecessary GPU memory.
    for state in flattened_osd["state"].values():
        for k in state.keys():
            state[k] = state[k].cpu()
    gc_cuda()

    optim.load_state_dict(fix_optim_state_dict(optim, flattened_osd))


def save_state_dict(
    checkpoint_dir: PathOrStr,
    fname: str,
    state_dict: Dict[str, Any],
    *,
    upload_to: Optional[str] = None,
    save_overwrite: bool = False,
    synchronize: bool = True,
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
    :param synchronize: If ``False``, don't do any distributed synchronization. Use this when only calling
        this function from a single rank.

    :raises FileExistsError: If the ``fname`` already exists within ``checkpoint_dir`` and ``save_overwrite=False``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    target_path = checkpoint_dir / fname
    if save_overwrite:
        target_path.unlink(missing_ok=True)
    elif target_path.is_file():
        raise FileExistsError(target_path)
    if synchronize:
        barrier()
    target_path.parent.mkdir(exist_ok=True, parents=True)
    if synchronize:
        barrier()
    torch.save(state_dict, target_path)
    if upload_to is not None:
        upload_target = f"{upload_to.rstrip('/')}/{fname}"
        log.info(f"Uploading {target_path} to {upload_target}...")
        upload(target_path, upload_target, save_overwrite=save_overwrite)


def load_state_dict(
    checkpoint_dir: PathOrStr,
    fname: str,
    *,
    local_cache: Optional[PathOrStr] = None,
    map_location: Optional[str] = None,
):
    """
    Load a regular state dict from the file ``fname`` within ``checkpoint_dir`` using :func:`torch.load()`.
    This can be used during distributed training or not.

    :param checkpoint_dir: A local or remote checkpoint directory.
    :param fname: The target file within the ``checkpoint_dir``. This should be a path relative to the ``checkpoint_dir``.
    :param local_cache: A local cache of the checkpoint directory. Use this when the ``checkpoint_dir`` is a
        remote "directory" but there might be a cached version of the same artifacts.

    :raises FileNotFoundError: If ``fname`` doesn't exist in the ``checkpoint_dir`` or the local cache.
    """
    if fname.endswith(".pt"):
        # Try safetensors version first.
        try:
            path = resource_path(
                str(checkpoint_dir).rstrip("/"), fname[:-2] + "safetensors", local_cache=local_cache
            )
            return safetensors_file_to_state_dict(path, map_location=map_location)
        except FileNotFoundError:
            pass

    path = resource_path(str(checkpoint_dir).rstrip("/"), fname, local_cache=local_cache)
    return torch.load(path, map_location=map_location, weights_only=False)


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
            # NOTE: we default to 1 thread here instead of whatever `default_thread_count()`
            # returns because uploading big checkpoint files with multiple threads causes
            # boto3 to fail in weird ways.
            thread_count=thread_count or 1,
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

            # Create the global S3 client up front to work around a threading issue in boto.
            if self.upload_to.startswith("s3://"):
                _get_s3_client("s3")
            elif self.upload_to.startswith("r2://"):
                _get_s3_client("r2")
            elif self.upload_to.startswith("weka://"):
                _get_s3_client("weka")

            with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                futures = []
                for fname in files_to_upload:
                    source = self.path / fname
                    target = f"{self.upload_to}/{fname}"
                    log.info(f"Uploading {source} to {target}...")
                    futures.append(executor.submit(upload, source, target, save_overwrite=self.save_overwrite))
                for f in as_completed(futures):
                    try:
                        f.result()
                    except BaseException:
                        # NOTE: we might get an error here that can't be pickled, which causes a different failure
                        # later when PyTorch tries to reduce that error across ranks. So here we just make
                        # sure we're raising a simple error type that can be pickled.
                        raise OLMoCheckpointError(f"Original error:\n{traceback.format_exc()}")
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
        self.thread_count = thread_count or default_thread_count()
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
        # Create the global S3 client up front to work around a threading issue in boto.
        if isinstance(self.path, str):
            if self.path.startswith("s3://"):
                _get_s3_client("s3")
            elif self.path.startswith("r2://"):
                _get_s3_client("r2")
            elif self.path.startswith("weka://"):
                _get_s3_client("weka")

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            read_item_content_futures = []
            for read_item in plan.items:
                read_item_content_futures.append(executor.submit(self._get_content_for_read, read_item))
            read_item_content_results = []
            for f in as_completed(read_item_content_futures):
                try:
                    read_item_content_results.append(f.result())
                except BaseException:
                    # NOTE: we might get an error here that can't be pickled, which causes a different failure
                    # later when PyTorch tries to reduce that error across ranks. So here we just make
                    # sure we're raising a simple error type that can be pickled.
                    raise OLMoCheckpointError(f"Original error:\n{traceback.format_exc()}")

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


class Checkpointer(metaclass=ABCMeta):
    def __init__(self, cfg: TrainConfig, thread_count: Optional[int] = None):
        self.cfg = cfg
        self.thread_count = thread_count or default_thread_count()

    @abstractmethod
    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        train_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Restores a checkpoint to the model and optimizer. Returns the remaining trainer state.
        """
        raise NotImplementedError

    def unshard_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Unshard a checkpoint.

        Note this is not marked abstract because child classes are not required to implemented this.
        """
        raise NotImplementedError

    @contextmanager
    def _temporary_wd(self, dir: PathOrStr) -> Generator[Path, None, None]:
        # Make sure checkpoint directory doesn't exist unless it's okay to overwrite it.
        checkpoint_dir = Path(dir)
        if not dir_is_empty(checkpoint_dir):
            if self.cfg.save_overwrite:
                if get_fs_local_rank() == 0:
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
            else:
                raise FileExistsError(checkpoint_dir)
        # No need to mkdir here since we'll directly replace the temporary directory with
        # this directory below.
        barrier()

        # Prepare temporary directory. We don't have to be as careful here, we can
        # just remove it if it already exists.
        checkpoint_dir_tmp = checkpoint_dir.with_name(checkpoint_dir.name + "-tmp")
        if get_fs_local_rank() == 0:
            shutil.rmtree(checkpoint_dir_tmp, ignore_errors=True)
            checkpoint_dir_tmp.mkdir(exist_ok=True, parents=True)

        # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
        # creating the temp directory from rank 0 might not be immediately
        # realized in the file systems of the other ranks.
        # So we wait here across all ranks until that tmp checkpoint directory is visible.
        wait_for(lambda: checkpoint_dir_tmp.exists(), "Waiting for checkpoint directory", timeout=10.0)

        barrier()

        # Yield temporary directory for `.save_checkpoint()` to use.
        yield checkpoint_dir_tmp

        barrier()

        # Finally if all went well replace the temporary directory with the actual
        # checkpoint directory.
        if get_fs_local_rank() == 0:
            # Replace temp directory with target checkpoint directory.
            try:
                checkpoint_dir_tmp.replace(checkpoint_dir)
            except FileNotFoundError:
                # Caught when another (file-system) local rank 0 has already replaced the tmp directory.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if not checkpoint_dir.exists():
                    raise

        # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
        # replacing the temp directory with the final directory from rank 0 might not be immediately
        # realized in the file systems of the other ranks.
        # So we wait here across all ranks until that final checkpoint directory is visible.
        wait_for(lambda: checkpoint_dir.exists(), "Waiting for checkpoint directory", timeout=10.0)

        barrier()

    def _save_config(self, dir: PathOrStr, *, upload_to: Optional[str] = None) -> None:
        if get_global_rank() == 0:
            log.info("Saving config...")
            self.cfg.save(config_path := Path(dir) / "config.yaml")
            if upload_to is not None:
                upload_target = f"{upload_to}/config.yaml"
                log.info(f"Uploading {config_path} to {upload_target}")
                upload(config_path, upload_target, save_overwrite=self.cfg.save_overwrite)


class FullCheckpointer(Checkpointer):
    """
    A :class:`Checkpointer` that saves a single full model and optimizer state dictionary.
    """

    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        with self._temporary_wd(dir) as checkpoint_dir:
            if isinstance(dist_model, FSDP):
                with FSDP.state_dict_type(
                    dist_model,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    # We'll write the model and optimizer state dicts individually to reduce (CPU) memory consumption.
                    # First the model state.
                    model_state_dict = dist_model.state_dict()
                    self._write_model_dict(
                        model_state_dict, checkpoint_dir, upload_to, save_overwrite=self.cfg.save_overwrite
                    )

                    # Then the optimizer state.
                    optim_state_dict = FSDP.optim_state_dict(dist_model, optim)
                    self._write_optim_dict(
                        optim_state_dict, checkpoint_dir, upload_to, save_overwrite=self.cfg.save_overwrite
                    )
            elif isinstance(dist_model, (DDP, SingleAccelerator)):
                # _write_model_dict and _write_optim_dict only write checkpoints for rank 0
                # First, get the model state dict from DDP wrapped model
                model_state_dict = dist_model.module.state_dict()
                self._write_model_dict(
                    model_state_dict, checkpoint_dir, upload_to, save_overwrite=self.cfg.save_overwrite
                )

                # Then get the optimizer state dict
                optim_state_dict = optim.state_dict()
                self._write_optim_dict(
                    optim_state_dict, checkpoint_dir, upload_to, save_overwrite=self.cfg.save_overwrite
                )
            else:
                log.info(
                    "`FullCheckpointer.save_checkpoint` only supported for FSDP, DDP, and SingleAccelerator distributed strategies!"
                )

            # Save trainer state.
            if get_global_rank() == 0:
                log.info("Saving trainer state...")
                save_state_dict(
                    checkpoint_dir,
                    "train.pt",
                    trainer_state,
                    upload_to=upload_to,
                    save_overwrite=self.cfg.save_overwrite,
                    synchronize=False,
                )
            # Save config.
            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(dist_model, FSDP):
            with FSDP.state_dict_type(
                dist_model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
                optim_state_dict_config=FullOptimStateDictConfig(rank0_only=False, offload_to_cpu=True),
            ):
                with torch.no_grad():
                    # fill everything with NaN, so we can check afterwards that every parameter has been restored
                    for module_name, module in dist_model.named_modules():
                        if not isinstance(module, FSDP):
                            continue
                        for param in module.params:
                            param.fill_(torch.nan)

                    # restore params from checkpoint
                    state_dict_to_load = load_state_dict(
                        load_path, "model.pt", local_cache=local_cache, map_location="cpu"
                    )
                    (
                        state_dict_to_load,
                        og_keys_to_new,
                    ) = dist_model._fsdp_wrapped_module._make_state_dict_compatible(state_dict_to_load)

                    for module_name, module in dist_model.named_modules():
                        if not isinstance(module, FSDP):
                            continue
                        for param in module.params:
                            assert param._is_flat_param
                            for fqn, spi in zip(param._fqns, param._shard_param_infos):
                                if not spi.in_shard:
                                    continue
                                key = f"{module_name}.{fqn}"
                                key = key.replace("_fsdp_wrapped_module.", "")
                                key = key.lstrip(".")
                                t = state_dict_to_load[key]
                                t = t.flatten()
                                param[spi.offset_in_shard : spi.offset_in_shard + spi.numel_in_shard].copy_(
                                    t[spi.intra_param_start_idx : spi.intra_param_end_idx + 1]
                                )

                    # make sure that every parameter has been restored
                    for module_name, module in dist_model.named_modules():
                        if not isinstance(module, FSDP):
                            continue
                        for param in module.params:
                            if torch.isnan(param).any():
                                raise ValueError(
                                    f"Module '{module_name}' contains NaNs, this is likely a bug restoring from full checkpoints"
                                )

                # Load optimizer state.
                if load_optimizer_state:
                    optim_state_dict_to_load = load_state_dict(
                        load_path, "optim.pt", local_cache=local_cache, map_location="cpu"
                    )
                    optim_state_dict_to_load = self._make_optim_state_dict_compatible(
                        optim_state_dict_to_load,
                        og_keys_to_new,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    barrier()
                    for turn in range(get_local_world_size()):
                        log.info("Loading optimizer state turn %d ...", turn)
                        if turn == get_local_rank():
                            load_fsdp_optim_state(dist_model, optim, optim_state_dict_to_load)
                            gc.collect()
                            torch.cuda.empty_cache()
                        barrier()
                    del optim_state_dict_to_load
        elif isinstance(dist_model, (DDP, SingleAccelerator)):
            # Load model state.
            with torch.no_grad():
                state_dict_to_load = load_state_dict(
                    load_path, "model.pt", local_cache=local_cache, map_location="cpu"
                )
                dist_model.module.load_state_dict(state_dict_to_load, strict=True)

            # Load optimizer state.
            if load_optimizer_state:
                optim_state_dict_to_load = load_state_dict(
                    load_path, "optim.pt", local_cache=local_cache, map_location="cpu"
                )
                optim.load_state_dict(optim_state_dict_to_load)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            barrier()
        else:
            raise NotImplementedError(
                "`FullCheckpointer.restore_checkpoint` only supported for FSDP, DDP, and SingleAccelerator distributed strategies!"
            )

        # Load other state.
        try:
            trainer_state = load_state_dict(load_path, "train.pt", local_cache=local_cache)
        except FileNotFoundError:
            # for backwards compatibility
            trainer_state = load_state_dict(load_path, "other.pt", local_cache=local_cache)
        barrier()
        return trainer_state

    def _write_model_dict(self, model_state_dict, checkpoint_dir, upload_to, save_overwrite):
        if get_global_rank() == 0:
            log.info("Saving model state...")
            save_state_dict(
                checkpoint_dir,
                "model.pt",
                model_state_dict,
                upload_to=upload_to,
                save_overwrite=save_overwrite,
                synchronize=False,
            )

        del model_state_dict
        barrier()

    def _write_optim_dict(self, optim_state_dict, checkpoint_dir, upload_to, save_overwrite):
        if get_global_rank() == 0:
            log.info("Saving optim state...")
            save_state_dict(
                checkpoint_dir,
                "optim.pt",
                optim_state_dict,
                upload_to=upload_to,
                save_overwrite=save_overwrite,
                synchronize=False,
            )

        del optim_state_dict
        barrier()

    def _make_optim_state_dict_compatible(
        self, optim_state_dict: Dict[str, Any], og_keys_to_new: Dict[str, Set[str]]
    ) -> Dict[str, Any]:
        # This state dict comes in two forms: one where the state keys are integers and one where the
        # keys are fully qualified parameter names. The latter case is easier to deal with here so we
        # first transform the integer key form into the FQN key form.
        if isinstance(optim_state_dict["param_groups"][0]["params"][0], int):
            id_to_fqn: Dict[int, str] = {}
            for group in optim_state_dict["param_groups"]:
                new_param_names = []
                for fqn, id in zip(group["param_names"], group["params"]):
                    fqn = fqn.replace("_fsdp_wrapped_module.", "")
                    id_to_fqn[id] = fqn
                    new_param_names.append(fqn)
                group["param_names"] = new_param_names
                group["params"] = new_param_names
            for id in list(optim_state_dict["state"].keys()):
                optim_state_dict["state"][id_to_fqn[id]] = optim_state_dict["state"].pop(id)
        else:
            # Otherwise we still want to clean up the param names to remove the "_fsdp_wrapped_module." prefix.
            for group in optim_state_dict["param_groups"]:
                group["param_names"] = [fqn.replace("_fsdp_wrapped_module.", "") for fqn in group["param_names"]]
                group["params"] = [fqn.replace("_fsdp_wrapped_module.", "") for fqn in group["params"]]
                assert group["param_names"] == group["params"]
            for key in list(optim_state_dict["state"].keys()):
                optim_state_dict["state"][key.replace("_fsdp_wrapped_module.", "")] = optim_state_dict[
                    "state"
                ].pop(key)

        # Now we can transform the state dict by renaming parameters according to `og_keys_to_new`.
        # First fix param names in the state.
        for og_key, new_keys in og_keys_to_new.items():
            og_state = optim_state_dict["state"].pop(og_key, None)
            if og_state is None:
                continue
            for i, new_key in enumerate(new_keys):
                if i == len(new_keys) - 1:
                    optim_state_dict["state"][new_key] = og_state
                else:
                    optim_state_dict["state"][new_key] = deepcopy(og_state)
        # Now fix param names in the param groups.
        for group in optim_state_dict["param_groups"]:
            og_names = group["params"]
            new_names = []
            for og_key in og_names:
                for new_key in og_keys_to_new[og_key]:
                    new_names.append(new_key)
            group["params"] = new_names
            group["param_names"] = new_names

        return optim_state_dict

    def load_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        device = device if device is not None else torch.device("cpu")
        model_state = load_state_dict(load_path, "model.pt", local_cache=local_cache, map_location=device)  # type: ignore
        optim_state = None
        if load_optimizer_state:
            optim_state = load_state_dict(load_path, "optim.pt", local_cache=local_cache, map_location=device)  # type: ignore
        return model_state, optim_state


class TorchNewStyleShardedCheckpointer(Checkpointer):
    """
    A sharded :class:`Checkpointer` that uses PyTorch's new distributed checkpointing functionality.
    """

    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        assert isinstance(
            dist_model, FSDP
        ), f"{self.__class__.__name__} is being called to save a model where `distributed_strategy` is not FSDP."
        with self._temporary_wd(dir) as checkpoint_dir:
            # Save model and optim state.
            save_fsdp_model_and_optim_state(
                checkpoint_dir,
                dist_model,
                optim,
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )

            # Save trainer state.
            log.info("Saving trainer state...")
            save_state_dict(
                checkpoint_dir,
                f"train/rank{get_global_rank()}.pt",
                trainer_state,
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )

            # Save config.
            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        # Load model and optimizer state in place.
        log.info("Loading model and optimizer state...")
        assert isinstance(
            dist_model, FSDP
        ), f"{self.__class__.__name__} is being called to load a model where `distributed_strategy` is not FSDP."

        load_fsdp_model_and_optim_state(
            load_path,
            dist_model,
            optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )

        # Load trainer state dict.
        log.info("Loading trainer state...")
        try:
            trainer_state = load_state_dict(
                load_path, f"train/rank{get_global_rank()}.pt", local_cache=local_cache
            )
        except FileNotFoundError:
            # Fall back to rank 0 train state.
            # This can happen when we're restoring a checkpoint with a different world size.
            trainer_state = load_state_dict(load_path, "train/rank0.pt", local_cache=local_cache)
        barrier()
        return trainer_state


class TorchLegacyShardedCheckpointer(Checkpointer):
    """
    A sharded :class:`Checkpointer` that just uses `torch.save()` with extra logic for handling FSDP model
    and optim state.

    The world size must be kept consistent when using this checkpointer.
    """

    def __init__(self, cfg: TrainConfig, thread_count: Optional[int] = None, use_shared_mem_impl: bool = False):
        super().__init__(cfg, thread_count)
        self.use_shared_mem_impl = use_shared_mem_impl

    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        assert isinstance(
            dist_model, FSDP
        ), f"{self.__class__.__name__} is being called to save a model where `distributed_strategy` is not FSDP."
        with self._temporary_wd(dir) as checkpoint_dir:
            with FSDP.state_dict_type(
                dist_model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
            ):
                state_dict = {
                    "model": dist_model.state_dict(),
                    "optim": FSDP.optim_state_dict(dist_model, optim),
                    **trainer_state,
                }
                save_state_dict(
                    checkpoint_dir,
                    f"rank{get_global_rank()}.pt",
                    state_dict,
                    upload_to=upload_to,
                    save_overwrite=self.cfg.save_overwrite,
                )

            # Save config.
            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        assert isinstance(
            dist_model, FSDP
        ), f"{self.__class__.__name__} is being called to load a model where `distributed_strategy` is not FSDP."
        with FSDP.state_dict_type(
            dist_model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            # Deserialize state dict.
            state_dict = load_state_dict(
                load_path, f"rank{get_global_rank()}.pt", local_cache=local_cache, map_location="cpu"
            )

            # Load model and optimizer state.
            log.info("Loading model state...")
            dist_model.load_state_dict(state_dict["model"])
            del state_dict["model"]
            if load_optimizer_state:
                log.info("Loading optimizer state...")
                load_fsdp_optim_state(dist_model, optim, state_dict["optim"])
            del state_dict["optim"]

        barrier()
        return state_dict

    def unshard_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        assert local_cache is None, "this method currently only supports local files"
        full_state_dict = self._unshard(load_path, device or torch.device("cpu"), skip_keys={"rng"})
        model_state = full_state_dict.pop("model")
        optim_state = full_state_dict.pop("optim")
        return (
            model_state,
            optim_state if load_optimizer_state else None,
            full_state_dict if load_trainer_state else None,
        )

    def _copy_sharded_tensors_to_shared_mem(self, state: Dict, world_size: int, rank: int, key: Tuple):
        key = tuple() if key is None else key
        if isinstance(state, (list, tuple, set)):
            for i, sub_state in enumerate(state):
                self._copy_sharded_tensors_to_shared_mem(sub_state, world_size, rank, key + (i,))
        elif isinstance(state, dict):
            for name in state.keys():
                self._copy_sharded_tensors_to_shared_mem(state[name], world_size, rank, key + (name,))
        elif isinstance(state, ShardedTensor):
            self._copy_sharded_tensor_to_shared_mem(state, world_size, rank, key)
            return
        else:
            return

    def _get_shard_placement_and_rank_sizes(
        self, shards_metadata: List[ShardMetadata], world_size: int
    ) -> Tuple[Dict[ShardMetadata, Tuple[int, int]], List[int]]:
        def shard_size(shard_md):
            return reduce((lambda x, y: x * y), shard_md.shard_sizes)  # type: ignore[attr-defined]

        rank_sizes = [0 for _ in range(world_size)]
        shard_placement: Dict[ShardMetadata, Tuple[int, int]] = {}
        for shard_md in shards_metadata:
            shard_rank = cast(_remote_device, shard_md.placement).rank()
            assert shard_rank is not None
            if shard_rank >= world_size:
                raise RuntimeError(f"Shard rank {shard_rank} exceeds world size {world_size}")

            shard_placement[shard_md] = (shard_rank, rank_sizes[shard_rank])
            rank_sizes[shard_rank] += shard_size(shard_md)

        return shard_placement, rank_sizes

    def _copy_sharded_tensor_to_shared_mem(
        self, sharded_tensor: ShardedTensor, world_size: int, rank: int, key: Tuple
    ) -> Any:
        shard0_md = sharded_tensor.metadata()
        shard_placement, rank_sizes = self._get_shard_placement_and_rank_sizes(
            shard0_md.shards_metadata, world_size
        )

        rank_size = rank_sizes[rank]
        assert rank_size >= 0
        if rank_size == 0:
            return

        assert shard0_md.tensor_properties.dtype == torch.float32, "Expected sharded tensor to be fp32"
        numpy_type = np.float32

        sharded_memory_name = "-".join(key + (str(rank),))

        shm = shared_memory.SharedMemory(
            create=True, size=rank_size * np.dtype(numpy_type).itemsize, name=sharded_memory_name
        )
        np_arr = np.ndarray((rank_size,), dtype=numpy_type, buffer=shm.buf)

        for local_shard in sharded_tensor.local_shards():
            shard_rank = cast(_remote_device, local_shard.metadata.placement).rank()
            assert shard_rank == rank

            src = local_shard.tensor.flatten()
            shard_offset = shard_placement[local_shard.metadata][1]

            np_arr[shard_offset : shard_offset + src.numel()] = src.numpy()

        shm.close()

    def _copy_sharded_data_to_shared_mem(self, world_size: int, shard_filepath: Path):
        shard_number = int(shard_filepath.name[4:-3])
        log.info("Starting unsharding shard number %d to shared memory", shard_number)

        with self._patch_sharded_tensor_load():
            shard = torch.load(shard_filepath, map_location="cpu")
            log.debug("Done loading shard number %d", shard_number)

        self._copy_sharded_tensors_to_shared_mem(
            shard, world_size, shard_number, (str(shard_filepath.parent).replace("/", "_"),)
        )
        log.info("Done unsharding shard number %d to shared memory", shard_number)

    def _unshard_using_sharded_mem(
        self, state: Any, world_size: int, device: torch.device, shard_dir: PathOrStr
    ) -> Any:
        return self._unshard_state_using_shared_mem(state, world_size, device, (str(shard_dir).replace("/", "_"),))

    def _unshard_state_using_shared_mem(
        self, state: Any, world_size: int, device: torch.device, key: Tuple
    ) -> Any:
        if isinstance(state, (list, tuple, set)):
            return state.__class__(
                self._unshard_state_using_shared_mem(sub_state, world_size, device, key + (i,))
                for i, sub_state in enumerate(state)
            )
        elif isinstance(state, dict):
            return {
                name: self._unshard_state_using_shared_mem(state[name], world_size, device, key + (name,))
                for name in state.keys()
            }
        elif isinstance(state, ShardedTensor):
            return self._unshard_tensor_using_shared_mem(state, world_size, device, key)
        elif isinstance(state, torch.Tensor):
            return state.to(device=device)
        else:
            return state

    def _unshard_tensor_using_shared_mem(
        self, sharded_tensor: ShardedTensor, world_size: int, device: torch.device, key: Tuple
    ) -> torch.Tensor:
        shard0_md = sharded_tensor.metadata()

        def shard_size(shard_md):
            return reduce((lambda x, y: x * y), shard_md.shard_sizes)  # type: ignore[attr-defined]

        shard_placement, rank_sizes = self._get_shard_placement_and_rank_sizes(
            shard0_md.shards_metadata, world_size
        )

        assert shard0_md.tensor_properties.dtype == torch.float32, "Expected sharded tensor to be fp32"
        numpy_type = np.float32

        out = torch.empty(
            *sharded_tensor.metadata().size, dtype=sharded_tensor.metadata().tensor_properties.dtype, device=device
        )
        dims = len(sharded_tensor.metadata().size)
        for shard_md, (rank, rank_offset) in shard_placement.items():
            if rank >= world_size:
                raise RuntimeError(f"Shard rank {rank} exceeds world size {world_size}")

            sharded_memory_name = "-".join(key + (str(rank),))
            shm = shared_memory.SharedMemory(name=sharded_memory_name)

            rank_size = rank_sizes[rank]
            assert rank_size >= 0
            if rank_size == 0:
                continue

            np_arr = np.ndarray((rank_size,), dtype=numpy_type, buffer=shm.buf)

            tensor = torch.from_numpy(np_arr)[rank_offset : rank_offset + shard_size(shard_md)]
            tensor = tensor.view(shard_md.shard_sizes)

            out_narrow_view = out
            for dim in range(dims):
                out_narrow_view = out_narrow_view.narrow(
                    dim,
                    shard_md.shard_offsets[dim],
                    shard_md.shard_sizes[dim],
                )

            out_narrow_view.copy_(tensor)

            shm.close()
            shm.unlink()

        return out

    @contextmanager
    def _patch_sharded_tensor_load(self):
        """
        Monkeypatch for torch's ShardedTensor, so we can unpickle without having torch.distributed set up.
        """

        def _rebuild_from_type_v2_monkey(func, new_type, args, state):
            ret = func(*args)
            if type(ret) is not new_type:
                ret = ret.as_subclass(new_type)

            # Shortcut the construction of ShardedTensor
            # This is in the top 5 of my worst hacks.
            if isinstance(ret, ShardedTensor):
                ret._local_shards, ret._metadata, _, ret._sharding_spec, ret._init_rrefs = state
                return ret

            # The rest of this function ought to be in the top 5 of somebody else's worst hacks.
            # Tensor does define __setstate__ even though it doesn't define
            # __getstate__. So only use __setstate__ if it is NOT the one defined
            # on Tensor
            if getattr(ret.__class__, "__setstate__", torch.Tensor.__setstate__) is not torch.Tensor.__setstate__:
                ret.__setstate__(state)
            else:
                ret = torch._utils._set_obj_state(ret, state)
            return ret

        original_rebuild_from_type_v2 = torch._tensor._rebuild_from_type_v2
        try:
            torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2_monkey
            yield
        finally:
            torch._tensor._rebuild_from_type_v2 = original_rebuild_from_type_v2

    def _unshard_using_shared_memory(
        self, input_dir: PathOrStr, device: torch.device, skip_keys: Optional[Set[str]] = None
    ):
        """
        This unsharding implementation consists of:

        1. Loading each shard on a separate process and copying their sharded tensors to shared memory.
        2. Loading 1 shard on the main process as a base unsharded object.
        3. Using the sharded tensors in shared memory to populate the base unsharded object.

        This implementation is an alternative to a prior implementation that instead loaded
        all shards using threads, because that implementation turned out to
        be extremely slow (e.g. 6+ hours) sometimes when the world size was 1024.
        The current implementation is slower than the old one in many scenarios,
        but is significantly faster in the above mentioned case (e.g. 30 minutes)
        if there are enough CPUs.

        We keep the other implementation since this once can be more unreliable,
        likely due to its dependence on a large amount of shared memory.
        """

        input_dir = Path(input_dir)
        skip_keys = skip_keys or set()

        shard_filepaths = list(input_dir.glob("rank*.pt"))
        world_size = len(shard_filepaths)
        if world_size == 0:
            raise RuntimeError("No shards found for unsharding")

        log.info("Number of shards: %d", world_size)
        shard_size_gb = shard_filepaths[0].stat().st_size / (1024 * 1024 * 1024)
        min_ram_required_estimate_gb = shard_size_gb * world_size
        log.info(
            "Shards are %.2fGB each, at least %.2fGB RAM is required", shard_size_gb, min_ram_required_estimate_gb
        )

        log.info("Copying sharded tensors to shared memory using multiple processes")
        # Copy sharded data to shared memory using multiple processes, so this process can load
        # from memory rather than disk. We spawn a new process instead of forking since shared memory
        # appears to get deleted when forked processes end for some reason.
        executor = ProcessPoolExecutor(
            mp_context=mp.get_context("spawn"), initializer=util.prepare_cli_environment
        )
        futures = []
        for shard_filepath in shard_filepaths:
            shard_rank = int(shard_filepath.name[4:-3])

            if shard_rank >= world_size:
                raise RuntimeError(
                    f"Shard rank {shard_rank} of file {shard_filepath} exceeds world size {world_size}"
                )

            futures.append(executor.submit(self._copy_sharded_data_to_shared_mem, world_size, shard_filepath))

        for f in as_completed(futures):
            f.result()
        executor.shutdown()

        log.info("Loading a shard on the main process to be unsharded state")
        with self._patch_sharded_tensor_load():
            state = torch.load(shard_filepaths[0], map_location="cpu")

        for key in skip_keys:
            if key in state:
                del state[key]

        log.info("Unsharding from %d shards ...", world_size)
        return self._unshard_using_sharded_mem(state, world_size, device, input_dir)

    def _unshard(self, input_dir: PathOrStr, device: torch.device, skip_keys: Optional[Set[str]] = None):
        if self.use_shared_mem_impl:
            return self._unshard_using_shared_memory(input_dir, device, skip_keys)

        input_dir = Path(input_dir)
        skip_keys = skip_keys or set()

        with self._patch_sharded_tensor_load():
            # We load in threads because it's faster.
            executor = ThreadPoolExecutor()
            shards_dict = {}
            for shard_name in input_dir.glob("rank*.pt"):
                log.info("Loading %s ...", shard_name)
                shard_number = int(shard_name.name[4:-3])  # shard names look like "rankXX.pt"
                shards_dict[shard_number] = executor.submit(torch.load, shard_name, map_location="cpu")
            shards = [None] * len(shards_dict)
            for rank, shard_future in shards_dict.items():
                shard = shard_future.result()
                for key in skip_keys:
                    if key in shard:
                        del shard[key]
                shards[rank] = shard
            assert all(shard is not None for shard in shards)
            executor.shutdown()
            del shards_dict

            log.info("Unsharding from %d shards ...", len(shards))

            unsharded_state_dict = self._unshard_object(shards, device=device)
            # At this point in time we need 2x memory :-(
            del shards

            return unsharded_state_dict

    def _unshard_object(self, os: List[Any], device: torch.device) -> Any:
        rank0_item = os[0]
        assert all(type(o) is type(rank0_item) for o in os)
        if isinstance(rank0_item, str):
            assert all(o == rank0_item for o in os)
            return rank0_item
        elif isinstance(rank0_item, (list, tuple, set)):
            assert all(len(o) == len(rank0_item) for o in os)
            return rank0_item.__class__(self._unshard_object(o, device=device) for o in zip(*os))
        elif isinstance(rank0_item, dict):
            assert all(o.keys() == rank0_item.keys() for o in os)
            return {key: self._unshard_object([o[key] for o in os], device=device) for key in rank0_item.keys()}
        elif isinstance(rank0_item, ShardedTensor):
            return self._gather(os, device=device)
        else:
            assert all(self._objects_are_equal(o, rank0_item) for o in os)
            return rank0_item

    def _gather(self, shards: List[ShardedTensor], device: torch.device) -> torch.Tensor:
        world_size = len(shards)
        shard0_md = shards[0].metadata()
        # Make sure all shards agree on the metadata
        assert all(shard.metadata() == shard0_md for shard in shards)
        # Make sure the nth shard expects to be the nth shard.
        assert all(
            shard_md.placement.rank() == rank  # type: ignore
            for rank, shard_md in enumerate(shard0_md.shards_metadata)
        )

        def shard_size(shard_md):
            return reduce((lambda x, y: x * y), shard_md.shard_sizes)  # type: ignore[attr-defined]

        rank_sizes = [0 for _ in range(world_size)]
        max_rank_size = 0
        shard_placement: Dict[ShardMetadata, Tuple[int, int]] = {}
        for shard_md in shard0_md.shards_metadata:
            shard_rank = cast(_remote_device, shard_md.placement).rank()
            assert shard_rank is not None

            shard_placement[shard_md] = (shard_rank, rank_sizes[shard_rank])
            rank_sizes[shard_rank] += shard_size(shard_md)
            max_rank_size = max(max_rank_size, rank_sizes[shard_rank])

        gather_list: List[torch.Tensor] = [torch.empty((max_rank_size,)) for _ in range(world_size)]

        datas = []
        with torch.no_grad():
            for shard in shards:
                data = torch.empty(max_rank_size)

                for local_shard in shard.local_shards():
                    src = local_shard.tensor.flatten()
                    shard_offset = shard_placement[local_shard.metadata][1]
                    data[shard_offset : shard_offset + src.numel()].copy_(src)

                datas.append(data)

        # torch.gather in a nutshell
        for rank, data in enumerate(datas):
            gather_list[rank].copy_(data)

        full_size = shard0_md.size
        out = torch.empty(*full_size, dtype=shard0_md.tensor_properties.dtype, device=device)
        dims = len(full_size)
        for shard_md in shard0_md.shards_metadata:
            rank, rank_offset = shard_placement[shard_md]
            tensor = gather_list[rank]
            tensor = tensor[rank_offset : rank_offset + shard_size(shard_md)]
            tensor = tensor.view(shard_md.shard_sizes)

            out_narrow_view = out
            for dim in range(dims):
                out_narrow_view = out_narrow_view.narrow(
                    dim,
                    shard_md.shard_offsets[dim],
                    shard_md.shard_sizes[dim],
                )

            out_narrow_view.copy_(tensor)

        return out

    def _objects_are_equal(self, a: Any, b: Any) -> bool:
        if type(a) is not type(b):
            return False
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        elif isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        else:
            return a == b


@dataclass
class _LocalShardedCheckpointerMetadata(BaseConfig):
    world_size: int = field(default_factory=get_world_size)


@dataclass
class _FlatParamShard:
    full_shape: torch.Size
    shard_offsets: Tuple[int, int]
    shard_data: Optional[torch.Tensor]

    def copy_into(self, full_tensor: torch.Tensor) -> None:
        assert self.shard_data is not None
        full_tensor_shard_view = full_tensor.view(-1)[self.shard_offsets[0] : self.shard_offsets[1] + 1]
        assert self.shard_data.shape == full_tensor_shard_view.shape
        full_tensor_shard_view.copy_(self.shard_data)


class LocalShardedCheckpointer(Checkpointer):
    """
    A sharded :class:`Checkpointer` that directly saves the local FSDP flat params data.
    The optimizer state is saved directly with `torch.save()` without reformatting via FSDP methods.

    The world size must be kept consistent when using this checkpointer. However, you can easily
    reconstruct a full unsharded model and/or optimizer state dictionary from a single Python process
    using :meth:`unshard_checkpoint()` (no distributed initialization required).
    """

    # These correspond to metadata attributes on `torch.distributed.fsdp.flat_param.FlatParameter`.
    _FLAT_PARAM_METADATA_TO_SAVE = (
        "_fqns",
        "_shard_param_offsets",
        "_shard_indices",
        "_numels",
        "_numels_with_padding",
        "_shapes",
        "_shard_numel_padded",
        "_shard_param_infos",
    )

    def _fsdp_modules(self, fsdp_model: FSDP) -> List[Tuple[str, FSDP]]:
        """
        Returns a list of FSDP modules with their FQN.
        """
        modules = []
        for name, module in fsdp_model.named_modules():
            if isinstance(module, FSDP):
                modules.append((name, module))
        return modules

    def _prepare_fsdp_model(self, fsdp_model: FSDP) -> None:
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        # TODO (epwalsh): I'm not sure if this is necessary, but this is what PyTorch does before saving/loading
        # an FSDP state dict through the built-in methods.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _lazy_init(fsdp_model, fsdp_model)

    def _fsdp_handles(self, fsdp_model: FSDP) -> List[FlatParamHandle]:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            return fsdp_model._handles  # type: ignore
        elif version.parse(torch.__version__) < version.parse("2.3.0"):
            # Handle could be None if the FSDP wrapper doesn't manage any parameters.
            if hasattr(fsdp_model, "_handle") and fsdp_model._handle is not None:
                return [fsdp_model._handle]  # type: ignore
            else:
                return []
        else:
            # Need to verify FSDP internals with newer versions.
            raise NotImplementedError

    @torch.no_grad()
    def _get_flat_param_state_to_save(self, fsdp_model: FSDP) -> Dict[str, Any]:
        self._prepare_fsdp_model(fsdp_model)
        module_data = []
        for module_fqn, fsdp_module in self._fsdp_modules(fsdp_model):
            handle_data = []
            for handle in self._fsdp_handles(fsdp_module):
                data: Dict[str, Any] = {}
                # This is a `FlatParameter` instance.
                # See `torch.distributed.fsdp.flat_param` for the API.
                flat_param = handle.flat_param
                data["flat_param.data"] = flat_param.detach()
                for key in self._FLAT_PARAM_METADATA_TO_SAVE:
                    if hasattr(flat_param, key):
                        data[f"flat_param.{key}"] = getattr(flat_param, key)
                handle_data.append(data)
            module_data.append({"handles": handle_data, "name": module_fqn})
        return {"modules": module_data}

    @torch.no_grad()
    def _load_flat_param_state(self, fsdp_model: FSDP, model_state: Dict[str, Any]):
        """Load the state produced from `self._get_flat_param_state_to_save()`."""
        self._prepare_fsdp_model(fsdp_model)
        fsdp_modules = self._fsdp_modules(fsdp_model)
        assert len(model_state["modules"]) == len(fsdp_modules)
        for (_, fsdp_module), module_data in zip(fsdp_modules, model_state["modules"]):
            handles = self._fsdp_handles(fsdp_module)
            assert len(handles) == len(module_data["handles"])
            for handle, data in zip(handles, module_data["handles"]):
                flat_param = handle.flat_param
                # Make sure metadata matches.
                for key in self._FLAT_PARAM_METADATA_TO_SAVE:
                    if hasattr(flat_param, key):
                        assert getattr(flat_param, key) == data[f"flat_param.{key}"]
                # Load the flat sharded data.
                flat_param.copy_(data["flat_param.data"])

    def _save_metadata(self, dir: PathOrStr, *, upload_to: Optional[str] = None) -> None:
        if get_fs_local_rank() == 0:
            log.info("Saving metadata...")
            metadata = _LocalShardedCheckpointerMetadata()
            metadata.save(metadata_path := Path(dir) / "metadata.yaml")
            if upload_to is not None and get_global_rank() == 0:
                upload_target = f"{upload_to}/metadata.yaml"
                log.info(f"Uploading {metadata_path} to {upload_target}")
                upload(metadata_path, upload_target, save_overwrite=self.cfg.save_overwrite)

    def _load_metadata(
        self, load_path: PathOrStr, *, local_cache: Optional[PathOrStr] = None
    ) -> _LocalShardedCheckpointerMetadata:
        metadata_path = resource_path(load_path, "metadata.yaml", local_cache=local_cache)
        return _LocalShardedCheckpointerMetadata.load(metadata_path)

    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        assert isinstance(
            dist_model, FSDP
        ), f"{self.__class__.__name__} is being called to save a model where `distributed_strategy` is not FSDP."

        with self._temporary_wd(dir) as checkpoint_dir:
            # Gather local FSDP flat params data to save.
            # We also save some flat param metadata like the corresponding fully qualified names (fqns)
            # of each original parameter so we can validate that the sharding is the same when loading
            # one of these checkpoints.
            log.info("Saving local FSDP flat params data...")
            save_state_dict(
                checkpoint_dir,
                f"model/rank{get_global_rank()}.pt",
                self._get_flat_param_state_to_save(dist_model),
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )

            # Save optimizer state.
            log.info("Saving local optimizer state...")
            save_state_dict(
                checkpoint_dir,
                f"optim/rank{get_global_rank()}.pt",
                optim.state_dict(),
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )

            # Save trainer state.
            log.info("Saving trainer state...")
            save_state_dict(
                checkpoint_dir,
                f"train/rank{get_global_rank()}.pt",
                trainer_state,
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )

            # Save metadata.
            self._save_metadata(checkpoint_dir, upload_to=upload_to)

            # Save config. We do this last b/c the presence of a config in a remote checkpoint
            # "directory" indicates that the folder is valid, as a opposed to a partially
            # uploaded checkpoint directory that failed before completing.
            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        # Load metadata and make sure checkpoint is compatible.
        metadata = self._load_metadata(load_path, local_cache=local_cache)
        assert metadata.world_size == get_world_size()

        # Load local FSDP flat param data.
        log.info("Loading local FSDP flat params data...")
        assert isinstance(
            dist_model, FSDP
        ), f"{self.__class__.__name__} is being called to load a model where `distributed_strategy` is not FSDP."

        model_state = load_state_dict(
            load_path, f"model/rank{get_global_rank()}.pt", local_cache=local_cache, map_location="cpu"
        )
        self._load_flat_param_state(dist_model, model_state)
        del model_state

        # Load local optim state.
        if load_optimizer_state:
            log.info("Loading local optimizer state...")
            optim_state = load_state_dict(
                load_path, f"optim/rank{get_global_rank()}.pt", local_cache=local_cache, map_location="cpu"
            )
            # HACK/TODO (epwalsh): When we use adaptive clipping we track the 'grad_norm_exp_avg' for every param
            # in every rank, and keep this in the optimizer state. But this causes issues when loading the
            # state since torch sees the state is non-empty for some params which would normally be empty,
            # and then assumes it should have all of the other state tensors for that param, which is doesn't.
            # So for now we just remove 'grad_norm_exp_avg' everywhere from the state, which resets that metric.
            # Not the end of the world but there's probably a better way around this without resetting
            # the metric.
            for param_id in list(optim_state["state"].keys()):
                state = optim_state["state"][param_id]
                if "grad_norm_exp_avg" in state:
                    del state["grad_norm_exp_avg"]
                if len(state) == 0:
                    del optim_state["state"][param_id]
            optim.load_state_dict(optim_state)
            del optim_state

        # Load local trainer state.
        log.info("Loading local trainer state...")
        trainer_state = load_state_dict(load_path, f"train/rank{get_global_rank()}.pt", local_cache=local_cache)
        barrier()
        return trainer_state

    def _iter_flat_param_shards(
        self, model_state: Dict[str, Any]
    ) -> Generator[Tuple[str, _FlatParamShard], None, None]:
        for module_data in model_state["modules"]:
            module_prefix = module_data["name"].replace("_fsdp_wrapped_module.", "")
            for handle in module_data["handles"]:
                flat_data = handle["flat_param.data"]
                if (num_padding := handle["flat_param._shard_numel_padded"]) > 0:
                    # If there's padding in the flat param it should be on the right.
                    assert (flat_data[-num_padding:] == 0).all()
                # NOTE: this changes depending on the torch version, but we don't do a version
                # check since we might be trying to unshard an old checkpoint that was stored
                # with a different torch version than we're currently running with.
                if "flat_param._shard_indices" in handle:
                    # torch <=2.0.1
                    param_start = handle["flat_param._shard_indices"][0]
                    current_flat_index = 0
                    for relative_fqn, full_shape, (offset_start, offset_end) in zip(
                        handle["flat_param._fqns"][param_start:],
                        handle["flat_param._shapes"][param_start:],
                        handle["flat_param._shard_param_offsets"],
                    ):
                        root_fqn = relative_fqn if not module_prefix else f"{module_prefix}.{relative_fqn}"
                        numel_shard = offset_end - offset_start + 1
                        flat_param_shard = _FlatParamShard(
                            full_shape=full_shape,
                            shard_offsets=(offset_start, offset_end),
                            shard_data=flat_data[current_flat_index : current_flat_index + numel_shard],
                        )
                        current_flat_index += numel_shard
                        yield root_fqn, flat_param_shard
                else:
                    # torch >=2.1.0
                    for relative_fqn, full_shape, shard_param_info in zip(
                        handle["flat_param._fqns"],
                        handle["flat_param._shapes"],
                        handle["flat_param._shard_param_infos"],
                    ):
                        if not shard_param_info.in_shard:
                            continue
                        root_fqn = relative_fqn if not module_prefix else f"{module_prefix}.{relative_fqn}"
                        flat_param_shard = _FlatParamShard(
                            full_shape=full_shape,
                            shard_offsets=(
                                shard_param_info.intra_param_start_idx,
                                shard_param_info.intra_param_end_idx,
                            ),
                            shard_data=flat_data[
                                shard_param_info.offset_in_shard : shard_param_info.offset_in_shard
                                + shard_param_info.numel_in_shard
                            ],
                        )
                        yield root_fqn, flat_param_shard

    def unshard_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        device = device or torch.device("cpu")
        metadata = self._load_metadata(load_path, local_cache=local_cache)

        # Gather paths model state, potentially downloading them.
        log.info("Gathering model state dicts...")
        model_state_paths = self._gather_state_dict_paths(
            load_path, "model", metadata.world_size, local_cache=local_cache
        )

        # Load model state dicts one-by-one, materializing and populating the full parameters as we go.
        log.info("Materializing full parameters...")
        full_model_state: Dict[str, torch.Tensor] = {}
        # We keep a copy of the flat param metadata minus the actual tensors so we can reconstruct
        # the full optimizer state below without having to reload the model state dicts.
        flat_params_data: Dict[int, Dict[str, _FlatParamShard]] = defaultdict(dict)
        for rank, path in enumerate(model_state_paths):
            log.info(f"Loading shards from rank {rank}...")
            model_state = torch.load(path, map_location="cpu")
            for root_fqn, flat_param_shard in self._iter_flat_param_shards(model_state):
                if root_fqn not in full_model_state:
                    log.info(
                        f"Materializing full parameter '{root_fqn}' with shape {flat_param_shard.full_shape}..."
                    )
                    assert flat_param_shard.shard_data is not None
                    full_model_state[root_fqn] = torch.empty(
                        flat_param_shard.full_shape, dtype=flat_param_shard.shard_data.dtype, device=device
                    )
                    # Fill with NaNs so we can validate that the whole parameter has been populated
                    # afterwards.
                    full_model_state[root_fqn].fill_(torch.nan)
                # Copy over the local shard to the relevant part of the full parameter.
                full_param = full_model_state[root_fqn]
                log.info(f"Loading rank {rank} shard for '{root_fqn}'...")
                flat_param_shard.copy_into(full_param)
                flat_params_data[rank][root_fqn] = replace(flat_param_shard, shard_data=None)

        log.info("Validating full parameters...")
        for key, tensor in full_model_state.items():
            if torch.isnan(tensor).any():
                raise ValueError(f"Parameter '{key}' contains NaNs, this is likely a bug with the unsharder")

        trainer_state: Optional[Dict[str, Any]] = None
        if load_trainer_state:
            trainer_state = load_state_dict(load_path, "train/rank0.pt", local_cache=local_cache)

        if not load_optimizer_state:
            return full_model_state, None, trainer_state

        log.info("Gathering optim state dicts...")
        optim_state_paths = self._gather_state_dict_paths(
            load_path, "optim", metadata.world_size, local_cache=local_cache
        )

        log.info("Materializing full optim state...")
        full_optim_state: Dict[str, Any] = {"state": defaultdict(dict)}
        fqn_to_id: Dict[str, int] = {}
        id_to_fqn: Dict[int, str] = {}
        for rank, path in enumerate(optim_state_paths):
            log.info(f"Loading sharded optim state from rank {rank}...")
            optim_state = torch.load(path, map_location="cpu")

            # Initialize param groups.
            # We assume parameter groups are the same across all ranks.
            # The only thing that differs across ranks is the state for each local sharded param.
            if "param_groups" not in full_optim_state:
                full_optim_state["param_groups"] = optim_state["param_groups"]
            else:
                assert full_optim_state["param_groups"] == optim_state["param_groups"]

            # Generate mapping of parameter FQNs to optimizer param IDs and vice-versa.
            if not fqn_to_id or not id_to_fqn:
                for group in full_optim_state["param_groups"]:
                    for fqn, id in zip(group["param_names"], group["params"]):
                        fqn = fqn.replace("_fsdp_wrapped_module.", "")
                        fqn_to_id[fqn] = id
                        id_to_fqn[id] = fqn

            # Iterate over local shard state and copy into the full state.
            for id, shard_state in optim_state["state"].items():
                fqn = id_to_fqn[id]
                flat_param_shard = flat_params_data[rank].get(fqn)  # type: ignore[assignment]
                full_state = full_optim_state["state"][id]
                for key, shard_value in shard_state.items():
                    assert isinstance(shard_value, torch.Tensor)
                    if shard_value.shape == torch.Size([]):
                        # Add singleton tensors directly to full state. These should be the same across
                        # all ranks.
                        assert key in ("step", "grad_norm_exp_avg")  # sanity check
                        if key not in full_state:
                            full_state[key] = shard_value.to(device)
                        else:
                            assert full_state[key] == shard_value
                    else:
                        # Otherwise we have a sharded param state.
                        # If the corresponding full param state hasn't been materialized yet, do so now.
                        assert flat_param_shard is not None, f"missing flat_params_data for {fqn} from rank {rank}"
                        if key not in full_state:
                            log.info(
                                f"Materializing full state '{key}' for '{fqn}' with shape {flat_param_shard.full_shape}..."
                            )
                            full_state[key] = torch.empty(
                                flat_param_shard.full_shape, dtype=shard_value.dtype, device=device
                            )
                        full_state_value = full_state[key]

                        # Copy over the local shard state to the relevant part of the full parameter state.
                        log.info(f"Loading rank {rank} shard state of '{key}' for '{fqn}'...")
                        replace(flat_param_shard, shard_data=shard_value).copy_into(full_state_value)

        # Lastly, clean up the parameter names in param groups.
        for group in full_optim_state["param_groups"]:
            group["param_names"] = [n.replace("_fsdp_wrapped_module.", "") for n in group["param_names"]]

        return full_model_state, full_optim_state, trainer_state

    def _get_state_dict_path(
        self,
        load_path: PathOrStr,
        state_dict_type: str,
        rank: int,
        *,
        local_cache: Optional[PathOrStr] = None,
        progress=None,
    ) -> Tuple[int, Path]:
        fname = f"{state_dict_type}/rank{rank}.pt"
        return rank, resource_path(str(load_path).rstrip("/"), fname, local_cache=local_cache, progress=progress)

    def _gather_state_dict_paths(
        self,
        load_path: PathOrStr,
        state_dict_type: str,
        world_size: int,
        *,
        local_cache: Optional[PathOrStr] = None,
    ) -> List[Path]:
        progress = get_progress_bar()
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = []
            for rank in range(world_size):
                future = executor.submit(
                    self._get_state_dict_path,
                    load_path,
                    state_dict_type,
                    rank,
                    local_cache=local_cache,
                    progress=progress,
                )
                futures.append(future)

            results: Dict[int, Path] = {}
            for future in as_completed(futures):
                rank, path = future.result()
                results[rank] = path

        return [results[rank] for rank in range(world_size)]


class OlmoCoreCheckpointer(Checkpointer):
    def save_checkpoint(
        self,
        dir: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        trainer_state: Dict[str, Any],
        *,
        upload_to: Optional[str] = None,
    ) -> None:
        from olmo_core.distributed.checkpoint import (  # type: ignore
            save_model_and_optim_state,
        )

        with self._temporary_wd(dir) as checkpoint_dir:
            log.info("Saving model and optim state...")
            if get_fs_local_rank() == 0:
                (checkpoint_dir / "model").mkdir(exist_ok=True, parents=True)
                (checkpoint_dir / "optim").mkdir(exist_ok=True, parents=True)
                (checkpoint_dir / "train").mkdir(exist_ok=True, parents=True)

            wait_for(
                lambda: (checkpoint_dir / "model").exists(), "Waiting for checkpoint model directory", timeout=10.0
            )
            wait_for(
                lambda: (checkpoint_dir / "optim").exists(), "Waiting for checkpoint optim directory", timeout=10.0
            )
            wait_for(
                lambda: (checkpoint_dir / "train").exists(), "Waiting for checkpoint train directory", timeout=10.0
            )

            local_files_created = save_model_and_optim_state(checkpoint_dir, dist_model, optim)
            if upload_to is not None:
                for path in local_files_created:
                    path = Path(path)
                    upload_target = f"{upload_to.rstrip('/')}/{path.relative_to(checkpoint_dir)}"
                    log.info(f"Uploading {path} to {upload_target}...")
                    upload(path, upload_target, save_overwrite=self.cfg.save_overwrite)

            log.info("Saving trainer state...")
            save_state_dict(
                checkpoint_dir,
                f"train/rank{get_global_rank()}.pt",
                trainer_state,
                upload_to=upload_to,
                save_overwrite=self.cfg.save_overwrite,
            )

            self._save_config(checkpoint_dir, upload_to=upload_to)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        dist_model: nn.Module,
        optim: Optimizer,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
    ) -> Dict[str, Any]:
        from olmo_core.distributed.checkpoint import (  # type: ignore
            load_model_and_optim_state,
        )

        log.info("Loading model and optim state...")
        load_model_and_optim_state(load_path, dist_model, optim if load_optimizer_state else None)

        log.info("Loading trainer state...")
        try:
            trainer_state = load_state_dict(
                load_path, f"train/rank{get_global_rank()}.pt", local_cache=local_cache
            )
        except FileNotFoundError:
            # Fall back to rank 0 train state.
            # This can happen when we're restoring a checkpoint with a different world size.
            trainer_state = load_state_dict(load_path, "train/rank0.pt", local_cache=local_cache)

        barrier()
        return trainer_state

    def unshard_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        from olmo_core.distributed.checkpoint import (  # type: ignore
            unshard_model_state,
            unshard_optim_state,
        )

        model_state = unshard_model_state(load_path, device=device)
        optim_state: Optional[Dict[str, Any]] = None
        train_state: Optional[Dict[str, Any]] = None
        if load_optimizer_state:
            optim_state = cast(Dict[str, Any], unshard_optim_state(load_path, device=device))
        if load_trainer_state:
            train_state = load_state_dict(load_path, "train/rank0.pt", local_cache=local_cache)
        return model_state, optim_state, train_state


def build_sharded_checkpointer(
    cfg: TrainConfig, *, name: Optional[ShardedCheckpointerType] = None, use_shared_mem_impl: bool = False
) -> Checkpointer:
    name = name or cfg.sharded_checkpointer
    if name == ShardedCheckpointerType.torch_new:
        return TorchNewStyleShardedCheckpointer(cfg)
    elif name == ShardedCheckpointerType.torch_legacy:
        return TorchLegacyShardedCheckpointer(cfg, use_shared_mem_impl=use_shared_mem_impl)
    elif name == ShardedCheckpointerType.local:
        return LocalShardedCheckpointer(cfg)
    elif name == ShardedCheckpointerType.olmo_core:
        return OlmoCoreCheckpointer(cfg)
    else:
        raise NotImplementedError(name)
