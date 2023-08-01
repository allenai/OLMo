import math
from bisect import bisect_right
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .config import OptimizerType, SchedulerType, TrainConfig

__all__ = ["LionW", "build_optimizer", "build_scheduler", "set_new_base_lr"]


class LionW(Optimizer):
    """Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py"""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def get_param_groups(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Separate parameters into weight decay and non weight decay groups.
    """
    from .model import LayerNormBase

    # Separate out parameters that we don't want to apply weight decay to, like norms and biases.
    decay = set()
    no_decay = set()
    all_params = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # NOTE: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times, but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn
            all_params[fpn] = p

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Linear):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (LayerNormBase, nn.LayerNorm, nn.Embedding)):
                no_decay.add(fpn)

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert decay
    assert no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    return [
        {"params": [all_params[pn] for pn in sorted(list(decay))]},
        {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]


def fix_optim_state_dict(optimizer: Optimizer, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure `state_dict`, which only have 1 param group, is compatible with the optimizer
    which may have two param groups (one for params with weight decay, the other for those without).
    """
    if len(state_dict["param_groups"]) == 1 and len(optimizer.param_groups) == 2:
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

        # Decay
        decay_param_group = {k: v for k, v in state_dict["param_groups"][0].items() if k != "params"}
        decay_param_group["params"] = optimizer.state_dict()["param_groups"][0]["params"]

        # No decay.
        no_decay_param_group = {k: v for k, v in state_dict["param_groups"][0].items() if k != "params"}
        no_decay_param_group["weight_decay"] = 0.0
        no_decay_param_group["params"] = optimizer.state_dict()["param_groups"][1]["params"]

        state_dict["param_groups"] = [decay_param_group, no_decay_param_group]
    return state_dict


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = (
        get_param_groups(model)
        if (cfg.optimizer.no_decay_norm_and_bias and cfg.optimizer.weight_decay > 0.0)
        else model.parameters()
    )
    if cfg.optimizer.name == OptimizerType.lionw:
        return LionW(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == OptimizerType.adam:
        return torch.optim.Adam(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == OptimizerType.adamw:
        return torch.optim.AdamW(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError


def build_scheduler(cfg: TrainConfig, optim: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    schedulers: List[torch.optim.lr_scheduler.LRScheduler] = []
    if cfg.scheduler.name == SchedulerType.cosine_with_warmup:
        milestones = [cfg.scheduler.t_warmup]
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=cfg.scheduler.alpha_f, end_factor=1.0, total_iters=cfg.scheduler.t_warmup
            )
        ]
        if cfg.scheduler.t_max is None:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                cfg.max_duration - cfg.scheduler.t_warmup,
                eta_min=cfg.optimizer.learning_rate * cfg.scheduler.alpha_f,
            )
            schedulers.append(cosine)
        else:
            milestones.append(cfg.scheduler.t_max)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                cfg.scheduler.t_max - cfg.scheduler.t_warmup,
                eta_min=cfg.optimizer.learning_rate * cfg.scheduler.alpha_f,
            )
            linear = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=cfg.scheduler.alpha_f,
                end_factor=cfg.scheduler.alpha_f**2,
                total_iters=cfg.max_duration - cfg.scheduler.t_max,
            )
            schedulers.append(cosine)
            schedulers.append(linear)
        return torch.optim.lr_scheduler.SequentialLR(optim, schedulers, milestones)
    elif cfg.scheduler.name == SchedulerType.inverse_sqrt_with_warmup:
        milestones = [cfg.scheduler.t_warmup]
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=cfg.scheduler.alpha_f, end_factor=1.0, total_iters=cfg.scheduler.t_warmup
            ),
            torch.optim.lr_scheduler.LambdaLR(optim, lambda step: 1.0 if step <= 0 else 1.0 / math.sqrt(step)),
        ]
        return torch.optim.lr_scheduler.SequentialLR(optim, schedulers, milestones)
    else:
        raise NotImplementedError


def set_new_base_lr(
    optim: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, new_base_lr: float
):
    """
    Set a new base learning rate in the optimizer and scheduler.
    """
    # Hack scheduler state to start with the new base LR.
    if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
        # Update 'base_lr' for all sub-schedulers.
        for sched in scheduler._schedulers:  # type: ignore
            sched.base_lrs = [new_base_lr] * len(sched.base_lrs)

        # Update '_last_lr' for current sub-scheduler.
        current_sched = scheduler._schedulers[bisect_right(scheduler._milestones, scheduler.last_epoch)]  # type: ignore
        if hasattr(current_sched, "_get_closed_form_lr"):
            current_sched._last_lr = current_sched._get_closed_form_lr()
        elif isinstance(current_sched, torch.optim.lr_scheduler.LambdaLR):
            current_sched._last_lr = current_sched.get_lr()  # type: ignore
        else:
            raise NotImplementedError
        scheduler._last_lr = current_sched.get_last_lr()  # type: ignore
    else:
        raise NotImplementedError

    # Update LR in optimizer.
    for param_group, new_lr in zip(optim.param_groups, scheduler.get_last_lr()):
        param_group["lr"] = new_lr
        param_group["initial_lr"] = new_base_lr
