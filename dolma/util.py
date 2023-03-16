import logging
import math
import os
import socket
import sys
import warnings
from typing import Tuple, Union

import rich
from composer.utils.dist import get_local_rank, get_node_rank
from rich.logging import RichHandler
from rich.text import Text

from .config import TrainConfig
from .exceptions import DolmaCliError, DolmaConfigurationError, DolmaError


def setup_logging():
    # We store these in variables because we don't want to create a bunch of syscalls every time
    # we write a log message. This also ensures that they always stay the same even if the host
    # goes crazy.
    node_rank = get_node_rank()
    local_rank = get_local_rank()
    hostname = socket.gethostname()
    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        record.node_rank = node_rank
        record.local_rank = local_rank
        record.hostname = hostname
        return record

    logging.setLogRecordFactory(log_record_factory)

    if (
        os.environ.get("DOLMA_NONINTERACTIVE", False)
        or os.environ.get("DEBIAN_FRONTEND", None) == "noninteractive"
        or not sys.stdout.isatty()
    ):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("{asctime}\t{hostname}:{local_rank}\t{levelname}\t{message}", style="{")
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        handler.setFormatter(formatter)
        logging.basicConfig(handlers=[handler], level=logging.INFO)
    else:
        logging.basicConfig(handlers=[RichHandler()], level=logging.INFO)

    logzio_token = os.environ.get("LOGZIO_TOKEN", None)
    if logzio_token is not None:
        from logzio.handler import LogzioHandler

        logging.getLogger().addHandler(LogzioHandler(logzio_token))

    logging.captureWarnings(True)


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
    elif issubclass(exctype, DolmaCliError):
        rich.print(f"[yellow]{value}[/]")
    elif issubclass(exctype, DolmaError):
        rich.print(Text(f"{exctype.__name__}:", style="red"), value)
    else:
        logging.getLogger().critical(
            "Uncaught %s: %s", exctype.__name__, value, exc_info=(exctype, value, traceback)
        )


def install_excepthook():
    sys.excepthook = excepthook


def filter_warnings():
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )
    # Filter composer warnings about loggers.
    warnings.filterwarnings(
        action="ignore",
        message="Specifying the ConsoleLogger via `loggers` is not recommended.*",
        module="composer.trainer.trainer",
    )
    # Torchvision warnings. We don't actually use torchvision at the moment
    # but composer imports it at some point and we see these warnings.
    warnings.filterwarnings(
        action="ignore",
        message="failed to load.*",
        module="torchvision.io.image",
    )


def set_env_variables():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_cli_environment():
    rich.reconfigure(width=max(rich.get_console().width, 180), soft_wrap=True)
    setup_logging()
    install_excepthook()
    filter_warnings()
    set_env_variables()


def clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


def calculate_batch_size_info(
    global_batch_size: int, device_microbatch_size: Union[int, str]
) -> Tuple[int, Union[str, int], Union[str, int]]:
    from composer.utils import dist

    if global_batch_size % dist.get_world_size() != 0:
        raise DolmaConfigurationError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_batch_size = global_batch_size // dist.get_world_size()
    if device_microbatch_size == "auto":
        device_grad_accum = "auto"
    elif isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_batch_size:
            warnings.warn(
                f"device_microbatch_size > device_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_batch_size}.",
                UserWarning,
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(device_batch_size / device_microbatch_size)
    else:
        raise DolmaConfigurationError(f"Not sure how to parse {device_microbatch_size=}")

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg: TrainConfig):
    from composer.utils import dist

    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg.global_train_batch_size, cfg.device_train_microbatch_size
    )
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_train_microbatch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if cfg.device_eval_batch_size is None:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        elif isinstance(cfg.device_train_microbatch_size, int):
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
        else:
            raise DolmaConfigurationError(
                f"Not sure how to parse device_train_microbatch_size={cfg.device_train_microbatch_size}"
            )
    return cfg
