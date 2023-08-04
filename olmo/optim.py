from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .config import OptimizerType, SchedulerType, TrainConfig

__all__ = [
    "LionW",
    "Scheduler",
    "CosWithWarmup",
    "InvSqrtWithWarmup",
    "MaxScheduler",
    "build_optimizer",
    "build_scheduler",
]


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


class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000) -> float:
        return initial_lr * (0.1 + 0.9 * min(step, warmup_steps) / warmup_steps)


@dataclass
class CosWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class InvSqrtWithWarmup(Scheduler):
    warmup_steps: int

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        del max_steps
        return initial_lr * sqrt(self.warmup_steps / max(self.warmup_steps, step))


@dataclass
class MaxScheduler(Scheduler):
    sched1: Scheduler
    sched2: Scheduler

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        return max(
            self.sched1.get_lr(initial_lr, step, max_steps), self.sched2.get_lr(initial_lr, step, max_steps)
        )


def get_param_groups(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Separate parameters into weight decay and non weight decay groups.
    """
    from .util import is_weight_decay_module

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
            elif pn.endswith("weight") and not is_weight_decay_module(m):
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


def build_scheduler(cfg: TrainConfig) -> Scheduler:
    sched_cfg = cfg.scheduler
    if cfg.scheduler.name == SchedulerType.cosine_with_warmup:
        return CosWithWarmup(warmup_steps=sched_cfg.t_warmup, alpha_f=sched_cfg.alpha_f, t_max=sched_cfg.t_max)
    elif cfg.scheduler.name == SchedulerType.inverse_sqrt_with_warmup:
        return InvSqrtWithWarmup(warmup_steps=sched_cfg.t_warmup)
    elif cfg.scheduler.name == SchedulerType.max_scheduler:
        return MaxScheduler(
            sched1=CosWithWarmup(
                warmup_steps=sched_cfg.t_warmup, alpha_f=sched_cfg.alpha_f, t_max=sched_cfg.t_max
            ),
            sched2=InvSqrtWithWarmup(warmup_steps=sched_cfg.t_warmup),
        )
    else:
        raise NotImplementedError
