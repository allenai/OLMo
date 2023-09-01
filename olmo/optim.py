import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim.optimizer import Optimizer as OptimizerBase

from .config import OptimizerType, SchedulerType, TrainConfig
from .util import get_default_device, is_distributed

__all__ = [
    "Optimizer",
    "LionW",
    "AdamW",
    "Scheduler",
    "CosWithWarmup",
    "InvSqrtWithWarmup",
    "MaxScheduler",
    "build_optimizer",
    "build_scheduler",
]


log = logging.getLogger(__name__)


class Optimizer(OptimizerBase):
    def get_pre_step_metrics(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        return self._collect_optim_param_metrics(module)

    def get_post_step_metrics(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        del module
        return {}

    def get_param_name(self, module: nn.Module, param: nn.Parameter) -> str:
        if not hasattr(self, "_param_to_name"):
            # NOTE (epwalsh): don't worry, this will not be included in `self.state_dict()`.
            self._param_to_name: Dict[nn.Parameter, str] = {}
            for name, param in module.named_parameters():
                self._param_to_name[param] = name
        return self._param_to_name[param]

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        del param
        return {}

    @torch.no_grad()
    def _collect_optim_param_metrics(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """
        A help method for collecting optimizer parameter metrics.
        If distributed training with FDSP, this implementation assumes `use_orig_params=True`.
        """
        # NOTE (epwalsh): during distributed training we're making an assumption that the order of
        # the param groups and the params within each group are the same across all ranks.
        # This is justified since we initialize the parameter groups in every rank by iterating over
        # `module.parameters()` or `module.named_modules()` / `module.named_parameters()`, each of which
        # provides a consistent order.
        #  For each parameter (with a gradient) we'll collect:
        # - min, max, avg, norm of the param itself
        # - min, max, avg, norm of the param's gradient
        # - min, max, avg, norm of any additional per-parameter optimizer state metrics returned from
        #   `self.get_state_for_param()`.
        # Afterwards we'll reduce these all over all ranks.
        per_param_min_metrics: List[torch.Tensor] = []
        per_param_max_metrics: List[torch.Tensor] = []
        per_param_sum_metrics: List[torch.Tensor] = []
        per_param_norm_metrics: List[torch.Tensor] = []
        per_param_numel_metrics: List[torch.Tensor] = []

        per_param_min_metric_names: List[str] = []
        per_param_max_metric_names: List[str] = []
        per_param_avg_metric_names: List[str] = []
        per_param_norm_metric_names: List[str] = []

        for group in self.param_groups:
            for p in group["params"]:
                name = self.get_param_name(module, p)
                state = self.get_state_for_param(p)
                sorted_state_keys = sorted(state.keys())
                tensors = [p, p.grad] + [state[key] for key in sorted_state_keys]
                prefixes = [f"param/{name}", f"grad/{name}"] + [f"{key}/{name}" for key in sorted_state_keys]

                # Get min, max, avg, and norm for all `tensors` associated with the parameter.
                for x, prefix in zip(tensors, prefixes):
                    # grad or state tensors could be none for params that have their shards completely on
                    # other ranks.
                    x = x if x is not None else torch.tensor([], device="cpu", dtype=torch.float32)
                    if x.numel() > 0:
                        x_abs = x.abs()
                        per_param_min_metrics.append(
                            x_abs.min().unsqueeze(0).to(device="cpu", dtype=torch.float32)
                        )
                        per_param_max_metrics.append(
                            x_abs.max().unsqueeze(0).to(device="cpu", dtype=torch.float32)
                        )
                        per_param_sum_metrics.append(x.sum().unsqueeze(0).to(device="cpu", dtype=torch.float32))
                        per_param_norm_metrics.append(
                            torch.linalg.vector_norm(x, 2.0, dtype=torch.float32).unsqueeze(0).to(device="cpu")
                        )
                        per_param_numel_metrics.append(
                            torch.tensor([x.numel()], device="cpu", dtype=torch.float32)
                        )
                    else:
                        per_param_min_metrics.append(
                            torch.tensor([float("inf")], device="cpu", dtype=torch.float32)
                        )
                        per_param_max_metrics.append(torch.tensor([0.0], device="cpu", dtype=torch.float32))
                        per_param_sum_metrics.append(torch.tensor([0.0], device="cpu", dtype=torch.float32))
                        per_param_norm_metrics.append(torch.tensor([0.0], device="cpu", dtype=torch.float32))
                        per_param_numel_metrics.append(torch.tensor([0.0], device="cpu", dtype=torch.float32))
                    per_param_min_metric_names.append(f"{prefix}.min")
                    per_param_max_metric_names.append(f"{prefix}.max")
                    per_param_avg_metric_names.append(f"{prefix}.avg")
                    per_param_norm_metric_names.append(f"{prefix}.norm")

        per_param_avg_metrics: List[torch.Tensor]
        if is_distributed() and isinstance(module, FullyShardedDataParallel):
            # Reduce mins.
            all_mins = torch.cat(per_param_min_metrics).to(get_default_device())
            dist.reduce(all_mins, 0, op=dist.ReduceOp.MIN)
            per_param_min_metrics = all_mins.to(device="cpu").split(1)
            # Reduce maxs.
            all_maxs = torch.cat(per_param_max_metrics).to(get_default_device())
            dist.reduce(all_maxs, 0, op=dist.ReduceOp.MAX)
            per_param_max_metrics = all_maxs.to(device="cpu").split(1)
            # Reduce sums.
            all_sums = torch.cat(per_param_sum_metrics).to(get_default_device())
            all_norms = torch.cat(per_param_norm_metrics).to(get_default_device()) ** 2.0
            all_numels = torch.cat(per_param_numel_metrics).to(get_default_device())
            all_sums_norms_numels = torch.cat(
                [all_sums.unsqueeze(0), all_norms.unsqueeze(0), all_numels.unsqueeze(0)], dim=0
            )
            dist.reduce(all_sums_norms_numels, 0, op=dist.ReduceOp.SUM)
            all_sums, all_norms, all_numels = all_sums_norms_numels.split(1)
            per_param_norm_metrics = (all_norms ** (0.5)).squeeze(0).to(device="cpu").split(1)
            # Get averages.
            # NOTE: could get infs for non-rank0 processes but that's okay.
            per_param_avg_metrics = (all_sums / all_numels).squeeze(0).to(device="cpu").split(1)
        else:
            per_param_avg_metrics = [x / n for x, n in zip(per_param_sum_metrics, per_param_numel_metrics)]

        all_metrics: Dict[str, torch.Tensor] = {}
        for metric_name, metric in zip(per_param_min_metric_names, per_param_min_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_max_metric_names, per_param_max_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_avg_metric_names, per_param_avg_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_norm_metric_names, per_param_norm_metrics):
            all_metrics[metric_name] = metric.squeeze(0)

        return all_metrics


class LionW(Optimizer):
    """
    Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

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
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None

    def get_post_step_metrics(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        update_total_dot_prod = self._update_total_dot_prod
        update_total_norm = self._update_total_norm
        signed_update_total_norm = self._signed_update_total_norm
        if update_total_dot_prod is None or update_total_norm is None or signed_update_total_norm is None:
            return {}

        if is_distributed() and isinstance(module, FullyShardedDataParallel):
            # Reduce total dot prod and norms across all ranks.
            update_total_norm = update_total_norm**2.0
            signed_update_total_norm = signed_update_total_norm**2.0
            # Reduce all together to avoid multiple communication calls.
            all_together = torch.stack([update_total_dot_prod, update_total_norm, signed_update_total_norm])
            # Only need the final result on rank0, since that's where we log from.
            dist.reduce(all_together, 0)
            update_total_dot_prod, update_total_norm, signed_update_total_norm = all_together
            update_total_norm = update_total_norm**0.5
            signed_update_total_norm = signed_update_total_norm**0.5

        update_cos_sim = update_total_dot_prod / torch.max(
            update_total_norm * signed_update_total_norm, torch.tensor(1e-8, device=get_default_device())
        )
        return {"update_cos_sim": update_cos_sim}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
        update_norms = []
        signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step weight decay
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
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                update_total_dot_prod = update_total_dot_prod.to(update.device)
                update_total_dot_prod += torch.tensordot(update, signed_update, dims=len(update.shape))
                update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))
                signed_update_norms.append(torch.linalg.vector_norm(signed_update, 2.0, dtype=torch.float32))

        # Compute cosine similarity between update and signed update.
        self._update_total_dot_prod = update_total_dot_prod.to(get_default_device())
        self._update_total_norm = torch.linalg.vector_norm(
            torch.stack(update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())
        self._signed_update_total_norm = torch.linalg.vector_norm(
            torch.stack(signed_update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())


class AdamW(torch.optim.AdamW, Optimizer):
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore


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


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> Optimizer:
    params = (
        get_param_groups(model)
        if (cfg.optimizer.no_decay_norm_and_bias and cfg.optimizer.weight_decay > 0.0)
        else model.parameters()
    )
    if isinstance(params, list):
        log.info(f"Constructing optimizer with {len(params)} param groups")
    else:
        log.info("Constructing optimizer with single param group")
    if cfg.optimizer.name == OptimizerType.lionw:
        return LionW(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == OptimizerType.adamw:
        return AdamW(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            eps=1e-5,
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
