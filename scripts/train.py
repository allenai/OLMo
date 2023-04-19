"""
This is the script used to train OLMo.

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

import logging
import os
import sys
from pathlib import Path
from typing import List, cast

import torch
from composer.callbacks import CheckpointSaver

from olmo import Olmo, TrainConfig
from olmo.exceptions import OlmoCliError, OlmoConfigurationError
from olmo.util import clean_opt, log_extra_field, prepare_cli_environment

log = logging.getLogger(__name__)


def main(cfg: TrainConfig) -> None:
    from composer import Trainer
    from composer.callbacks import LRMonitor, OptimizerMonitor, SpeedMonitor
    from composer.core import Callback
    from composer.loggers import WandBLogger
    from composer.loggers.logger_destination import LoggerDestination
    from composer.utils import dist, get_device, reproducibility
    from composer.utils.dist import get_global_rank

    from olmo.composer import (
        ComposerOlmoLM,
        OlmoConsoleLogger,
        build_algorithm,
        build_dataloader,
        build_evaluator,
        build_optimizer,
        build_scheduler,
        update_batch_size_info,
    )

    # Run name.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "train-llm")
    log_extra_field("run_name", cfg.run_name)

    cfg.model.precision = cfg.precision

    if get_global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OlmoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    # Set seed.
    reproducibility.seed_all(cfg.seed)

    # Initialize process group.
    dist.initialize_dist(get_device(None))

    # Update batch size info.
    update_batch_size_info(cfg)
    assert isinstance(cfg.device_train_batch_size, int)
    if get_global_rank() == 0:
        log.info(
            f"Using per-device training batch size of {cfg.device_train_batch_size} "
            f"for global batch size of {cfg.global_train_batch_size}"
        )

    # Initialize the model.
    olmo_model = Olmo(cfg.model)
    if get_global_rank() == 0:
        log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
        log.info(
            f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}",
        )

    # Compile it if necessary.
    if cfg.compile is not None:
        compile_kwargs = cfg.compile.asdict()
        # As far as duck typing is concerned, this is still an Olmo object.
        olmo_model = cast(Olmo, torch.compile(olmo_model, **compile_kwargs))

    # Optimizer.
    optimizer = build_optimizer(olmo_model, **cfg.optimizer.asdict())

    # Scheduler.
    scheduler = build_scheduler(cfg.scheduler)

    # Dataset / data loader.
    train_loader = build_dataloader(cfg.data, cfg.model, cfg.device_train_batch_size)

    # Evaluators.
    evaluators = [build_evaluator(eval_config, cfg.model) for eval_config in cfg.evaluators]

    # Algorithms.
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.algorithms or {}).items()
        if algorithm_cfg is not None
    ]

    # Callbacks.
    callbacks: List[Callback] = [
        SpeedMonitor(**cfg.speed_monitor.asdict()),
        LRMonitor(),
        OptimizerMonitor(log_optimizer_metrics=False),
    ]

    # Loggers.
    loggers: List[LoggerDestination] = [OlmoConsoleLogger(log_interval=cfg.console_log_interval)]
    if cfg.wandb is not None:
        loggers.append(WandBLogger(init_kwargs={"config": cfg.asdict(exclude=["wandb"])}, **cfg.wandb.asdict()))

    # Wrap model into composer model.
    composer_model = ComposerOlmoLM(olmo_model)
    del olmo_model

    # Trainer.
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=composer_model,
        train_dataloader=train_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        eval_dataloader=evaluators,
        eval_interval=cfg.eval_interval,
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
    log.info(
        f"Local rank: {dist.get_local_rank()}/{dist.get_local_world_size()}, "
        f"global rank: {dist.get_global_rank()}/{dist.get_world_size()}, "
        f"training on {device_id}"
    )

    if not cfg.dry_run and cfg.load_path is None:
        log.info("Saving pre-train checkpoint...")
        # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever)
        for callback in trainer.state.callbacks:
            if isinstance(callback, CheckpointSaver):
                callback._save_checkpoint(trainer.state, trainer.logger)

    if not cfg.dry_run:
        log.info("Starting training...")
        trainer.fit()

    trainer.close()

    if cfg.dry_run:
        log.info("Dry run complete")
    else:
        log.info("Training complete")


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
