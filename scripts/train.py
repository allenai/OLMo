"""
This is the script used to train DOLMA.

There is one required positional argument, the path to a YAML :class:`TrainConfig`.
Following the YAML path, you could pass any number of options to override
values in the :class:`TrainConfig`.

For example, to override :data:`TrainConfig.model.n_layers` to 5, pass ``--model.n_layers=5``:

```bash
python scripts/train.py train_config.yaml --model.n_layers=5
```

For distributed training you should run this script via the 'composer' CLI. For example:

```bash
composer scripts/train.py train_config.yaml ...
```
"""
import datetime
import os
import sys
from typing import Any, Dict, Union, List

import torch

from composer.devices import Device
from composer.utils import get_device
from composer.utils.dist import get_world_size, log, get_global_rank, get_local_rank, get_local_world_size, \
    get_node_rank

from dolma import SchedulerConfig, TrainConfig
from dolma.data import build_dataloader
from dolma.exceptions import DolmaCliError, DolmaConfigurationError
from dolma.util import clean_opt, echo, prepare_cli_environment, update_batch_size_info


def build_scheduler(cfg: SchedulerConfig):
    from composer.optim.scheduler import (
        ConstantWithWarmupScheduler,
        CosineAnnealingWithWarmupScheduler,
        LinearWithWarmupScheduler,
    )

    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise DolmaConfigurationError(f"Not sure how to build scheduler: {cfg.name}")


def build_algorithm(name: str, kwargs: Dict[str, Any]):
    from composer import algorithms

    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    #  elif name == 'alibi':
    #      return algorithms.Alibi(**kwargs)
    elif name == "fused_layernorm":
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == "low_precision_layernorm":
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")

def replacement_initialize_dist(device: Union[str, Device], timeout: float = 300.0):
    """
    This does exactly the same as Mosaic's ``initialize_dist()`` function, but it
    uses a file on NFS to initialize the process group.
    """
    import torch.distributed as dist

    # If device is string, get corresponding composer.devices.Device object
    device_obj = get_device(device)
    timeout_timedelta = datetime.timedelta(seconds=timeout)

    if get_world_size() > 1 and not dist.is_available():
        raise RuntimeError('When the world size is > 1, ``torch.distributed`` must be used. However, it is '
                           'not available in your installation of PyTorch. Please install or build PyTorch '
                           'with distributed support.')

    if dist.is_initialized():
        if dist.get_backend() != device_obj.dist_backend.lower():
            raise RuntimeError(f'The requested backend ({device_obj.dist_backend}) differs from the backend '
                               f'of the current process group ({dist.get_backend()}). If you '
                               'wish to change backends, please restart the python process.')
        return

    # If any of these variables are set, and they do not match the single rank defaults,
    # then do not automatically configure distributed. There are no reasonable defaults to infer
    # for the other variables. Instead, let torch.dist error on an incomplete configuration.

    # If none of these variables are set, or some are set but they match the single rank defaults,
    # then fill the rest in.

    dist_env_var_defaults = {
        'NODE_RANK': '0',
        'WORLD_SIZE': '1',
        'LOCAL_WORLD_SIZE': '1',
        'RANK': '0',
        'LOCAL_RANK': '0',
    }

    log.debug(
        'Initializing torch.dist: global_rank=%d, local_rank=%d, world_size=%d, local_world_size=%d, node_rank=%d',
        get_global_rank(),
        get_local_rank(),
        get_world_size(),
        get_local_world_size(),
        get_node_rank(),
    )

    dist_env_vars_match_defaults = all(os.environ.get(k, v) == v for (k, v) in dist_env_var_defaults.items())

    if dist_env_vars_match_defaults:
        # Fill in the remaining single-rank variables
        os.environ.update(dist_env_var_defaults)
        dist.init_process_group(device_obj.dist_backend, store=dist.HashStore(), world_size=1, rank=0)
    else:
        filename = os.environ.get("TORCH_DISTRIBUTED_INIT_FILE")
        dist.init_process_group(
            device_obj.dist_backend,
            init_method="file://" + filename,
            world_size=get_world_size(),
            rank=get_global_rank(),
            timeout=timeout_timedelta)


def main(cfg: TrainConfig) -> None:
    from composer import Trainer
    from composer.core import Callback
    from composer.loggers import WandBLogger
    from composer.loggers.logger_destination import LoggerDestination
    from composer.utils import dist, get_device, reproducibility

    from dolma.composer import ComposerDolmaGPT, DolmaConsoleLogger, SpeedMonitorMFU

    echo.info("Configuration:", cfg, rank_zero_only=True)

    reproducibility.seed_all(cfg.seed)
    replacement_initialize_dist(get_device(None))

    # Run name.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "llm")

    # Update batch size info.
    update_batch_size_info(cfg)
    assert isinstance(cfg.device_train_batch_size, int)
    echo.info(
        f"Using per-device training batch size of {cfg.device_train_batch_size} "
        f"for global batch size of {cfg.global_train_batch_size}",
        rank_zero_only=True,
    )

    # Model.
    model = ComposerDolmaGPT(cfg.model)
    echo.info(f"Total number of parameters: {model.model.num_params():,d}", rank_zero_only=True)
    echo.info(
        f"Number of non-embedding parameters: {model.model.num_params(include_embedding=False):,d}",
        rank_zero_only=True,
    )

    # Optimizer.
    optimizer = model.model.configure_optimizer(**cfg.optimizer.asdict())

    # Scheduler.
    scheduler = build_scheduler(cfg.scheduler)

    # Dataset / data loader.
    train_loader = build_dataloader(cfg, cfg.device_train_batch_size)

    # Algorithms.
    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in (cfg.algorithms or {}).items()]

    # Callbacks.
    callbacks: List[Callback] = [SpeedMonitorMFU()]

    # Loggers.
    loggers: List[LoggerDestination] = [DolmaConsoleLogger(log_interval=cfg.console_log_interval)]
    if cfg.wandb is not None:
        loggers.append(WandBLogger(**cfg.wandb.asdict()))

    # Trainer.
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        #  eval_dataloader=evaluators,
        #  eval_interval=cfg.eval_interval,
        #  eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        max_duration=cfg.max_duration,
        precision=cfg.precision,
        device_train_microbatch_size=cfg.device_train_microbatch_size,
        fsdp_config=cfg.fsdp_config,
        save_folder=cfg.save_folder,
        save_interval=cfg.save_interval,
        save_num_checkpoints_to_keep=cfg.save_num_checkpoints_to_keep,
        save_overwrite=cfg.save_overwrite,
        load_path=cfg.load_path,
        load_weights_only=cfg.load_weights_only,
        callbacks=callbacks,
        loggers=loggers,
        algorithms=algorithms,
        progress_bar=False,
        log_to_console=False,
        console_log_interval=cfg.console_log_interval,
    )

    device_id = trainer.state.device.name.upper()
    if device_id == "GPU":
        device_id += f" {torch.cuda.current_device()}"
    echo.info(
        f"Local rank: {dist.get_local_rank()}/{dist.get_local_world_size()}, "
        f"global rank: {dist.get_global_rank()}/{dist.get_world_size()}, "
        f"training on {device_id}"
    )

    if not cfg.dry_run:
        trainer.fit()

    trainer.close()

    if cfg.dry_run:
        echo.success("Dry run complete", rank_zero_only=True)
    else:
        echo.success("Training complete", rank_zero_only=True)


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise DolmaCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
