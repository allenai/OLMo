import functools
import logging
import math
import os
import re
import socket
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    TypeVar,
    Union,
)

import rich
import torch
import torch.distributed as dist
import torch.nn as nn
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.text import Text
from rich.traceback import Traceback
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .aliases import PathOrStr
from .config import LogFilterType
from .exceptions import OlmoCliError, OlmoError

_log_extra_fields: Dict[str, Any] = {}


def log_extra_field(field_name: str, field_value: Any) -> None:
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value


def setup_logging(log_filter_type: LogFilterType = LogFilterType.rank0_only) -> None:
    """
    :param rank0_only: INFO and below messages will only be emitted on the rank0 process.
    """
    log_extra_field("hostname", socket.gethostname())
    if is_distributed():
        log_extra_field("node_rank", get_node_rank())
        log_extra_field("local_rank", get_local_rank())
        log_extra_field("global_rank", get_global_rank())
    else:
        log_extra_field("node_rank", 0)
        log_extra_field("local_rank", 0)
        log_extra_field("global_rank", 0)

    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        for field_name, field_value in _log_extra_fields.items():
            setattr(record, field_name, field_value)
        return record

    logging.setLogRecordFactory(log_record_factory)

    handler: logging.Handler
    if (
        os.environ.get("OLMo_NONINTERACTIVE", False)
        or os.environ.get("DEBIAN_FRONTEND", None) == "noninteractive"
        or not sys.stdout.isatty()
    ):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s\t%(hostname)s:%(local_rank)s\t%(name)s:%(lineno)s\t%(levelname)s\t%(message)s"
        )
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        handler.setFormatter(formatter)
    else:
        handler = RichHandler()

    def rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "global_rank", 0) == 0:
            return 1
        else:
            return 0

    def local_rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "local_rank", 0) == 0:
            return 1
        else:
            return 0

    filter = None
    if log_filter_type == LogFilterType.rank0_only:
        filter = rank0_filter
    elif log_filter_type == LogFilterType.local_rank0_only:
        filter = local_rank0_filter  # type: ignore
    else:
        raise ValueError(log_filter_type)

    if filter is not None:
        handler.addFilter(filter)  # type: ignore
    logging.basicConfig(handlers=[handler], level=logging.INFO)

    logzio_token = os.environ.get("LOGZIO_TOKEN", None)
    if logzio_token:
        from logzio.handler import LogzioHandler

        logzio_handler = LogzioHandler(logzio_token)
        if filter is not None:
            logzio_handler.addFilter(filter)  # type: ignore
        logging.getLogger().addHandler(logzio_handler)

    logging.captureWarnings(True)


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
    elif issubclass(exctype, OlmoCliError):
        rich.get_console().print(f"[yellow]{value}[/]", highlight=False)
    elif issubclass(exctype, OlmoError):
        rich.get_console().print(Text(f"{exctype.__name__}:", style="red"), value, highlight=False)
    else:
        logging.getLogger().critical(
            "Uncaught %s: %s", exctype.__name__, value, exc_info=(exctype, value, traceback)
        )


def install_excepthook():
    sys.excepthook = excepthook


def filter_warnings():
    # Filter internal deprecation warnings from torch
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="TypedStorage is deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="Please use DTensor instead.*",
    )
    # Torchvision warnings. We don't actually use torchvision.
    warnings.filterwarnings(
        action="ignore",
        message="failed to load.*",
        module="torchvision.io.image",
    )


def set_env_variables():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_cli_environment(log_filter_type: Optional[LogFilterType] = None):
    if log_filter_type is None:
        log_filter_type = LogFilterType(os.environ.get("LOG_FILTER_TYPE", "rank0_only"))
    rich.reconfigure(width=max(rich.get_console().width, 180), soft_wrap=True)
    setup_logging(log_filter_type=log_filter_type)
    install_excepthook()
    filter_warnings()
    set_env_variables()


def clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


class RichHandler(logging.Handler):
    """
    A simplified version of rich.logging.RichHandler from
    https://github.com/Textualize/rich/blob/master/rich/logging.py
    """

    def __init__(
        self,
        *,
        level: Union[int, str] = logging.NOTSET,
        console: Optional[Console] = None,
        markup: bool = False,
    ) -> None:
        super().__init__(level=level)
        self.console = console or rich.get_console()
        self.highlighter = NullHighlighter()
        self.markup = markup

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record.msg, "__rich__") or hasattr(record.msg, "__rich_console__"):
                self.console.print(record.msg)
            else:
                msg: Any = record.msg
                if isinstance(record.msg, str):
                    msg = self.render_message(record=record, message=record.getMessage())
                renderables = [
                    self.get_time_text(record),
                    self.get_level_text(record),
                    self.get_location_text(record),
                    msg,
                ]
                if record.exc_info is not None:
                    tb = Traceback.from_exception(*record.exc_info)  # type: ignore
                    renderables.append(tb)
                self.console.print(*renderables)
        except Exception:
            self.handleError(record)

    def render_message(self, *, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        return message_text

    def get_time_text(self, record: logging.LogRecord) -> Text:
        log_time = datetime.fromtimestamp(record.created)
        time_str = log_time.strftime("[%Y-%m-%d %X]")
        return Text(time_str, style="log.time", end=" ")

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        level_text = Text.styled(level_name.ljust(8), f"logging.level.{level_name.lower()}")
        level_text.style = "log.level"
        level_text.end = " "
        return level_text

    def get_location_text(self, record: logging.LogRecord) -> Text:
        name_and_line = f"{record.name}:{record.lineno}" if record.name != "root" else "root"
        text = f"[{name_and_line}, rank={record.local_rank}]"  # type: ignore
        return Text(text, style="log.path")


def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


T = TypeVar("T")


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


def is_distributed() -> bool:
    if "LOCAL_RANK" in os.environ:
        return True
    else:
        return False


def get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK") or (get_global_rank() - get_local_rank()) // get_local_world_size())


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE") or 1)


def get_global_rank() -> int:
    return int(os.environ.get("RANK") or dist.get_rank())


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def peak_gpu_memory(reset: bool = False) -> Optional[float]:
    """
    Get the peak GPU memory usage in MB across all ranks.
    Only rank 0 will get the final result.
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if dist.is_available() and dist.is_initialized():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return peak_mb


def syncronize_flag(flag: bool, device: torch.device) -> bool:
    if dist.is_available() and dist.is_initialized():
        flag_tensor = torch.tensor(flag, device=device)
        dist.broadcast(flag_tensor, 0)
        return flag_tensor.item()  # type: ignore
    else:
        return flag


def wait_on(condition: Callable[[], bool], description: str, timeout: float = 10.0):
    """Wait on the condition function to return True."""
    start_time = time.monotonic()
    while not condition():
        time.sleep(0.5)
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"{description} timed out")


def is_url(path: PathOrStr) -> bool:
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None


def resource_path(folder: PathOrStr, fname: str) -> PathOrStr:
    if is_url(folder):
        from cached_path import cached_path

        return cached_path(f"{folder}/{fname}")
    else:
        return Path(folder) / fname


def file_size(path: PathOrStr) -> int:
    """
    Get the size of a local or remote file in bytes.
    """
    if is_url(path):
        from urllib.parse import urlparse

        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            return _gcs_file_size(parsed.netloc, parsed.path)
        elif parsed.scheme == "s3":
            return _s3_file_size(parsed.netloc, parsed.path)
        elif parsed.scheme == "file":
            return file_size(str(path).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"file size not implemented for '{parsed.scheme}' files")
    else:
        return os.stat(path).st_size


def upload(source: PathOrStr, target: str, save_overwrite: bool = False):
    """Upload source file to a target location on GCS or S3."""
    from urllib.parse import urlparse

    source = Path(source)
    assert source.is_file()
    parsed = urlparse(target)
    if parsed.scheme == "gs":
        _gcs_upload(source, parsed.netloc, parsed.path, save_overwrite=save_overwrite)
    elif parsed.scheme == "s3":
        _s3_upload(source, parsed.netloc, parsed.path, save_overwrite=save_overwrite)
    else:
        raise NotImplementedError(f"Upload not implemented for '{parsed.scheme}' scheme")


def _gcs_file_size(bucket_name: str, key: str) -> int:
    from google.api_core.exceptions import NotFound
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError("gs://{bucket_name}/{key}")
    assert blob.size is not None
    return blob.size


def _gcs_upload(source: Path, bucket_name: str, key: str, save_overwrite: bool = False):
    from google.cloud import storage as gcs

    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    if not save_overwrite and blob.exists():
        raise FileExistsError(f"gs://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it.")
    blob.upload_from_filename(source)


def _s3_upload(source: Path, bucket_name: str, key: str, save_overwrite: bool = False):
    import boto3
    from botocore.exceptions import ClientError

    s3_client = boto3.client("s3")
    if not save_overwrite:
        try:
            s3_client.head_object(Bucket=bucket_name, Key=key)
            raise FileExistsError(f"gs://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it.")
        except ClientError as e:
            if int(e.response["Error"]["Code"]) != 404:
                raise
    s3_client.upload_file(source, bucket_name, key)


def _s3_file_size(bucket_name: str, key: str) -> int:
    import boto3
    from botocore.exceptions import ClientError

    s3_client = boto3.client("s3")
    try:
        return s3_client.head_object(Bucket=bucket_name, Key=key)["ContentLength"]
    except ClientError as e:
        if int(e.response["Error"]["Code"]) != 404:
            raise
        raise FileNotFoundError("s3://{bucket_name}/{key}")


class GradParamNorms(NamedTuple):
    grad_norm: torch.Tensor
    param_norm: torch.Tensor
    #  grad_param_angle: torch.Tensor  # TODO


def fsdp_clip_grads_and_get_norms(module: FSDP, max_norm: float, norm_type: float = 2.0) -> GradParamNorms:
    """
    Clip the gradient norms of all parameters in an FSDP module. The norm
    is computed over all parameters' gradients as viewed as a single vector,
    and the gradients are modified in-place.

    Adapted from PyTorch's `FullyShardedDataParallel.clip_grad_norm_()` method to also return
    the parameter norm.

    :param max_norm: max allowed norm of the gradients.
    :param norm_type: type of the p-norm. Can be ``inf`` for infinity norm.

    Returns the gradient norm and parameter norm.
    """
    import torch.distributed.fsdp._traversal_utils as traversal_utils
    from torch.distributed.fsdp._common_utils import TrainingState

    module._assert_state(TrainingState.IDLE)

    # NOTE: Skipped check if every FSDP instance uses `NO_SHARD` since we don't use that.

    # Collect parameters are gradients.
    sharded_params: Set[nn.Parameter] = set()
    nonsharded_params: Set[nn.Parameter] = set()  # `NO_SHARD` or not FSDP-managed
    grads: List[torch.Tensor] = []
    for handle in traversal_utils._get_fsdp_handles(module):
        target_set = sharded_params if handle.uses_sharded_strategy else nonsharded_params
        if handle._use_orig_params:
            for param in handle.flat_param._params:
                target_set.add(param)
                if param.grad is not None:
                    grads.append(param.grad)
        else:
            target_set.add(handle.flat_param)
            if handle.flat_param.grad is not None:
                grads.append(handle.flat_param.grad)
    for param in module.parameters():
        not_fsdp_managed = param not in sharded_params and param not in nonsharded_params
        if not_fsdp_managed:
            nonsharded_params.add(param)
            if param.grad is not None:
                grads.append(param.grad)

    # Compute total norms.
    total_grad_norm = _get_total_norm(module, sharded_params, nonsharded_params, norm_type, for_gradients=True)
    total_param_norm = _get_total_norm(module, sharded_params, nonsharded_params, norm_type, for_gradients=False)

    # Clip gradients.
    clip_coef = max_norm / (total_grad_norm + 1e-6)
    # Multiplying by the clamped coefficient is meaningless when it is
    # equal to 1, but it avoids the host-device sync that would result from
    # `if clip_coef < 1`
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.detach().mul_(clip_coef_clamped.to(grad.device, grad.dtype))

    # NOTE: skipped case where `len(grads) == 0`
    assert grads

    total_grad_norm_dtype = functools.reduce(
        lambda dtype1, dtype2: torch.promote_types(dtype1, dtype2),
        [grad.dtype for grad in grads],
    )
    total_param_norm_dtype = functools.reduce(
        lambda dtype1, dtype2: torch.promote_types(dtype1, dtype2),
        [param.dtype for param in sharded_params] + [param.dtype for param in nonsharded_params],
    )
    return GradParamNorms(
        grad_norm=total_grad_norm.to(total_grad_norm_dtype), param_norm=total_param_norm.to(total_param_norm_dtype)
    )


def _get_grad_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
) -> torch.Tensor:
    """
    Returns the gradient norm of parameters, where the gradients
    are viewed as a single vector. The returned norm is in FP32 even if
    parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.

    Adapted from `torch.distributed.fsdp.fully_sharded_data_parallel._get_grad_norm()`.
    """
    grads = [param.grad for param in params if param.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)

    grad_dtypes = {grad.dtype for grad in grads}
    if len(grad_dtypes) != 1:
        raise ValueError(f"Requires uniform dtype across all gradients but got {grad_dtypes}")

    # Compute the gradient norm in FP32, where we treat the gradients as a single vector.
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32) for grad in grads],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm


def _get_param_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
) -> torch.Tensor:
    """
    Returns the norm of all parameters that have gradients, where the params
    are viewed as a single vector. The returned norm is in FP32 even if
    parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.

    Adapted from `torch.distributed.fsdp.fully_sharded_data_parallel._get_grad_norm()`.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        return torch.tensor(0.0)

    param_dtypes = {param.dtype for param in params_with_grad}
    if len(param_dtypes) != 1:
        raise ValueError(f"Requires uniform dtype across all parameters but got {param_dtypes}")

    # Compute the norm in FP32, where we treat the params as a single vector.
    param_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(param.data.detach(), norm_type, dtype=torch.float32)
                for param in params_with_grad
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return param_norm


def _get_total_norm(
    module: FSDP,
    sharded_params: Iterable[nn.Parameter],
    nonsharded_params: Iterable[nn.Parameter],
    norm_type: float,
    for_gradients: bool = True,
) -> torch.Tensor:
    if for_gradients:
        local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(module.compute_device)
        local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(module.compute_device)
    else:
        local_sharded_norm = _get_param_norm(sharded_params, norm_type).to(module.compute_device)
        local_nonsharded_norm = _get_param_norm(nonsharded_params, norm_type).to(module.compute_device)

    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=module.process_group)
    else:
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=module.process_group)
        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        total_norm += local_nonsharded_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    if module.cpu_offload.offload_params:
        total_norm = total_norm.cpu()

    return total_norm
