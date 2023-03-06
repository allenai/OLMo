import math
import os
import sys
import warnings
from typing import Any, Tuple, Union

import rich

from .config import TrainConfig
from .exceptions import DolmaCliError, DolmaConfigurationError, DolmaError


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    from rich import print
    from rich.traceback import Traceback

    if isinstance(value, DolmaCliError):
        print(f"[yellow]{value}[/]", file=sys.stderr)
    elif isinstance(value, DolmaError):
        print(f"[b red]{exctype.__name__}:[/] {value}", file=sys.stderr)
    else:
        print(Traceback.from_exception(exctype, value, traceback), file=sys.stderr)


def install_excepthook():
    sys.excepthook = excepthook


def filter_warnings():
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )
    # Torchvision warnings. We don't actually use torchvision at the moment
    # but composer imports it at some point and we see these warnings.
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=".*Failed to load.*",
    )


def set_env_variables():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_cli_environment():
    install_excepthook()
    filter_warnings()
    set_env_variables()


def clean_opt(arg: str) -> str:
    arg = arg.strip("-").replace("-", "_")
    if "=" not in arg:
        arg = f"{arg}=True"
    return arg


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


class echo:
    print = rich.print

    @classmethod
    def info(cls, *msgs: Any):
        cls.print("[blue]\N{information source}[/]", *msgs)

    @classmethod
    def success(cls, *msgs: Any):
        cls.print("[green]\N{check mark}[/]", *msgs)
