import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from math import cos, pi, sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.optimizer import Optimizer as OptimizerBase

try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

from . import LayerNormBase, RMSLayerNorm
from .config import OptimizerType, SchedulerConfig, SchedulerType, TrainConfig
from .torch_util import get_default_device, is_distributed

__all__ = [
    "Optimizer",
    "LionW",
    "AdamW",
    "Scheduler",
    "CosWithWarmup",
    "LinearWithWarmup",
    "InvSqrtWithWarmup",
    "MaxScheduler",
    "ConstantScheduler",
    "CosLinearEnvelope",
    "BoltOnWarmupScheduler",
    "build_optimizer",
    "build_scheduler",
]

# TODO: start adamw: beta1, beta2 experiments (warmup and cosine for batch size 4M)
# TODO: add muon and schedule free adamW, start experiments for adamw, schedule free and muon
# TODO: add ademamix, adan
# TODO: read schedule free paper, surprising agreement papers, ch 1-4 optimization course


log = logging.getLogger(__name__)


class Optimizer(OptimizerBase):
    def __init__(self, *args, record_update_metrics: bool = False, selective_updates: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._record_update_metrics = record_update_metrics
        self._collecting_metrics = False
        self._selective_updates = selective_updates

    def _clean_param_name(self, name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    @torch.no_grad()
    def clip_grads_and_collect_metrics(
        self,
        global_step: int,
        collect_param_metrics: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Clips gradients for every group that has the field `max_grad_norm`.
        At the same time collect metrics for each parameter and its gradient.
        """
        self._collecting_metrics = collect_param_metrics
        device = get_default_device() if device is None else device

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

        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)

        #######################################################################
        # part 1: collect metrics locally
        #######################################################################
        for group in self.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                # Always need to collect the norm of gradients for clipping, even if we're not collecting
                # other metrics.
                tensors: List[Optional[torch.Tensor]] = [p.grad]
                prefixes: List[str] = [f"grad/{name}"]
                if collect_param_metrics:
                    state = self.get_state_for_param(p)
                    sorted_state_keys = sorted([k for k in state.keys()])
                    tensors.extend([p] + [state[key] for key in sorted_state_keys])
                    prefixes.extend([f"param/{name}"] + [f"{key}/{name}" for key in sorted_state_keys])
                assert len(tensors) == len(prefixes)

                # Get min, max, avg, and norm for all `tensors` associated with the parameter.
                for x, prefix in zip(tensors, prefixes):
                    # grad or state tensors could be none for params that have their shards completely on
                    # other ranks.
                    if x is not None and x.numel() > 0:
                        if collect_param_metrics:
                            x_abs = x.abs()
                            per_param_min_metrics.append(x_abs.min().unsqueeze(0).to(dtype=torch.float32))
                            per_param_max_metrics.append(x_abs.max().unsqueeze(0).to(dtype=torch.float32))
                            per_param_sum_metrics.append(x.sum().unsqueeze(0).to(dtype=torch.float32))
                            per_param_numel_metrics.append(
                                torch.tensor([x.numel()], device=device, dtype=torch.float32)
                            )
                        per_param_norm_metrics.append(
                            torch.linalg.vector_norm(x, 2.0, dtype=torch.float32).unsqueeze(0)
                        )
                    else:
                        if collect_param_metrics:
                            per_param_min_metrics.append(
                                torch.tensor([float("inf")], device=device, dtype=torch.float32)
                            )
                            per_param_max_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_sum_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_numel_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                        per_param_norm_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                    if collect_param_metrics:
                        per_param_min_metric_names.append(f"{prefix}.min")
                        per_param_max_metric_names.append(f"{prefix}.max")
                        per_param_avg_metric_names.append(f"{prefix}.avg")
                    per_param_norm_metric_names.append(f"{prefix}.norm")

        assert (
            len(per_param_min_metrics)
            == len(per_param_min_metric_names)
            == len(per_param_max_metrics)
            == len(per_param_max_metric_names)
            == len(per_param_sum_metrics)
            == len(per_param_numel_metrics)
            == len(per_param_avg_metric_names)
        )
        assert len(per_param_norm_metrics) == len(per_param_norm_metric_names)

        def is_grad_norm_metric(metric_name: str) -> bool:
            return metric_name.startswith("grad/") and metric_name.endswith(".norm")

        #######################################################################
        # part 2: reduce metrics over ranks
        #######################################################################
        param_group_sharded = False
        for group in self.param_groups:
            param_group_sharded = param_group_sharded or group.get("sharded", False)

        total_grad_norm: torch.Tensor
        per_param_avg_metrics: List[torch.Tensor] = []
        if is_distributed() and param_group_sharded:
            # Reduce metrics across all ranks. Note that we can use a `reduce` for most cases
            # instead of an `all_reduce`, but we need `all_reduce` for norms so that all ranks
            # get the right value for gradient norms so they can clip correctly.
            # Reduce mins.
            if per_param_min_metrics:
                all_mins = torch.cat(per_param_min_metrics).to(device)
                dist.reduce(all_mins, dst_rank, op=dist.ReduceOp.MIN, group=process_group)
                per_param_min_metrics = all_mins.split(1)
            # Reduce maxs.
            if per_param_max_metrics:
                all_maxs = torch.cat(per_param_max_metrics).to(device)
                dist.reduce(all_maxs, dst_rank, op=dist.ReduceOp.MAX, group=process_group)
                per_param_max_metrics = all_maxs.split(1)
            # Reduce sums or just norms.
            all_norms = torch.cat(per_param_norm_metrics).to(device) ** 2.0
            if per_param_sum_metrics and per_param_numel_metrics:
                all_sums = torch.cat(per_param_sum_metrics).to(device)
                all_numels = torch.cat(per_param_numel_metrics).to(device)
                all_sums_norms_numels = torch.cat(
                    [all_sums.unsqueeze(0), all_norms.unsqueeze(0), all_numels.unsqueeze(0)], dim=0
                )
                dist.all_reduce(all_sums_norms_numels, op=dist.ReduceOp.SUM, group=process_group)
                all_sums, all_norms, all_numels = all_sums_norms_numels.split(1)
                # Get averages.
                # NOTE: could get infs for non-rank0 processes but that's okay.
                per_param_avg_metrics = (all_sums / all_numels).squeeze(0).split(1)
            else:
                dist.all_reduce(all_norms, op=dist.ReduceOp.SUM, group=process_group)
            grad_norm_metric_mask = torch.tensor(
                [float(is_grad_norm_metric(n)) for n in per_param_norm_metric_names], device=all_norms.device
            )
            total_grad_norm = (all_norms * grad_norm_metric_mask).sum() ** 0.5
            per_param_norm_metrics = (all_norms ** (0.5)).squeeze(0).split(1)
        else:
            total_grad_norm = (
                torch.cat(
                    [
                        m
                        for m, n in zip(per_param_norm_metrics, per_param_norm_metric_names)
                        if is_grad_norm_metric(n)
                    ]
                )
                ** 2.0
            ).sum() ** 0.5
            per_param_avg_metrics = [x / n for x, n in zip(per_param_sum_metrics, per_param_numel_metrics)]

        assert len(per_param_avg_metrics) == len(per_param_avg_metric_names)

        # Collect all metrics into a single dict.
        all_metrics: Dict[str, torch.Tensor] = {}
        if collect_param_metrics:
            for metric_name, metric in zip(per_param_min_metric_names, per_param_min_metrics):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(per_param_max_metric_names, per_param_max_metrics):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(per_param_avg_metric_names, per_param_avg_metrics):
                all_metrics[metric_name] = metric.squeeze(0)

        for metric_name, metric in zip(per_param_norm_metric_names, per_param_norm_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        all_metrics["total_grad_norm"] = total_grad_norm

        #######################################################################
        # part 3: clip grads
        #######################################################################
        num_grads_clipped = 0
        num_eligible_grads = 0
        for group in self.param_groups:
            if (max_norm_ratio := group.get("max_grad_norm_ratio")) is not None:
                num_clipped = self._do_adaptive_clipping(
                    group, max_norm_ratio, global_step, all_metrics, collect_param_metrics=collect_param_metrics
                )
            elif (max_norm := group.get("max_grad_norm")) is not None:
                num_clipped = self._do_global_fixed_clipping(
                    group, max_norm, all_metrics, collect_param_metrics=collect_param_metrics
                )
            else:
                # No clipping needed.
                continue
            num_eligible_grads += len(group["params"])
            if num_clipped is not None:
                num_grads_clipped += num_clipped

        if collect_param_metrics:
            if num_eligible_grads > 0:
                clipping_rate = torch.tensor(num_grads_clipped / num_eligible_grads, device="cpu")
            else:
                clipping_rate = torch.tensor(0.0, device="cpu")
            all_metrics["clipping_rate"] = clipping_rate

        # total_grad_norm is computed at all steps, even when collect_param_metrics is set to False
        return all_metrics

    @torch.no_grad()
    def _do_adaptive_clipping(
        self,
        group: Dict[str, Any],
        max_norm_ratio: float,
        global_step: int,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        """
        Do adaptive gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device() if device is None else device
        num_grads_clipped = 0
        # We'll use the bigger of beta1 and beta2 to update the exponential average of the norm of
        # the gradient (a scalar), not to be confused with the exponential average of the gradient.
        # TODO (epwalsh): handle optimizers that don't have betas.
        beta1, beta2 = group["betas"]
        beta = max(beta1, beta2)
        for name, p in zip(group["param_names"], group["params"]):
            name = self._clean_param_name(name)
            grad_norm = all_metrics.get(f"grad/{name}.norm")
            if grad_norm is None:
                continue

            # Get or initialize the exponential average of grad norm.
            # TODO: The way we have it right now, every rank tracks the `grad_norm_exp_avg` of every parameter,
            # even parameters for which the corresponding local shard is empty. This has the potential to
            # cause some issues with the optimizer, as we ran into with https://github.com/allenai/LLM/pull/372.
            # So we should consider changing how we do this at some point so that we don't add any state
            # to parameters for which the local shard is empty. That would probably add extra distributed
            # communication, at least on steps where we have to log (i.e. when `collect_param_metrics=True`).
            state = self.state[p]
            grad_norm_exp_avg = state.get("grad_norm_exp_avg")
            if grad_norm_exp_avg is None:
                grad_norm_exp_avg = grad_norm.clone().to(device)
                # We don't want to add anything to `state` until `state` has been initialized, otherwise
                # this will crash some optimizers which rely on checking `len(state)`. The downside here
                # is that we won't start tracking `grad_norm_exp_avg` until the 2nd training step.
                if global_step > 1:
                    state["grad_norm_exp_avg"] = grad_norm_exp_avg

            max_allowed_norm = max_norm_ratio * grad_norm_exp_avg
            clip_coef = max_allowed_norm / (grad_norm + 1e-6)

            # Clip the gradients and update the exponential average.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))

            # Update the exponential average of the norm of the gradient with the clipped norm of the gradient.
            grad_norm_exp_avg.lerp_((grad_norm * clip_coef_clamped).to(grad_norm_exp_avg.device), 1 - beta)
            # Alternative: update with the *unclipped* norm of the gradient.
            #  grad_norm_exp_avg.lerp_(grad_norm.to(grad_norm_exp_avg.device), 1 - beta)

            if collect_param_metrics:
                # Can't avoid host-device sync here.
                if clip_coef_clamped < 1.0:
                    num_grads_clipped += 1
                all_metrics[f"grad_norm_exp_avg/{name}"] = grad_norm_exp_avg
        return num_grads_clipped if collect_param_metrics else None

    @torch.no_grad()
    def _do_global_fixed_clipping(
        self,
        group: Dict[str, Any],
        max_norm: float,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        """
        Do global fixed gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device() if device is None else device
        total_grad_norm = all_metrics["total_grad_norm"]
        clip_coef = max_norm / (total_grad_norm.to(device) + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        num_grads_clipped: Optional[int] = None
        if collect_param_metrics:
            # Can't avoid host-device sync here.
            if clip_coef_clamped < 1.0:
                num_grads_clipped = len(group["params"])
        for p in group["params"]:
            # Clip the gradients.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))
        return num_grads_clipped

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        del module, process_group
        return {}

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        del param
        return {}


class SGD(Optimizer):
    """
        Option to add Polyak and Nesterov momentum.
    """

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


class ScheduleFreeSGD(Optimizer):

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


class MuonAdamW(Optimizer):
    """
        Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads,
        and scalar or vector parameters should be optimized using AdamW instead. Muon provides an internal AdamW
        for this so you don't have to use an extra optimizer.

        Codebase designed to handle single optimizer, merging muon and adamw here for different param groups.
        Will probably fail at checkpointing, restoring and logging a bunch of optimizer states.

        Code taken and modified from: https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    def __init__(
        self,
        params,
        muon_lr=0.02,
        muon_weight_decay=0.01,
        muon_momentum=0.95,
        nesterov=True,
        ns_steps=5,
        rank=0,
        world_size=1,
        adam_lr=1e-3,
        adam_betas=(0.9, 0.999),
        eps=1e-8,
        adam_weight_decay=0.01,
        amsgrad=False,
        vocab_size=50304,
        *args,
        record_update_metrics = False,
        selective_updates = False,
        **kwargs
    ):
        self.rank = rank
        self.world_size = world_size

        muon_defaults = dict(
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            optimizer_type='muon',
        )
        adamw_defaults = dict(
            lr=adam_lr,
            betas=adam_betas,
            eps=eps,
            weight_decay=adam_weight_decay,
            amsgrad=amsgrad,
            optimizer_type='adamw',
        )

        param_groups = []

        for old_group in params:
            if 'weight_decay' in old_group:
                del adamw_defaults['weight_decay']
                param_groups.append({**old_group, **adamw_defaults})
            else:
                # for each old_group, split into muon and adamw params
                adamw_params = {'params': [], 'param_names': []}
                muon_params = {'params': [], 'param_names': []}

                # TODO: create different groups for the same sized params for efficient distributed communication
                for pn, p in zip(old_group['param_names'], old_group['params']):
                    if p.ndim < 2 or p.shape[0] == vocab_size or p.shape[1] == vocab_size:
                        adamw_params['params'].append(p)
                        adamw_params['param_names'].append(pn)
                    else:
                        muon_params['params'].append(p)
                        muon_params['param_names'].append(pn)

                adamw_params = {
                    **adamw_params,
                    **adamw_defaults,
                    'sharded': old_group['sharded'],
                    'max_grad_norm': old_group['max_grad_norm'],
                    'max_grad_norm_ratio': old_group['max_grad_norm_ratio'],
                }
                param_groups.append(adamw_params)

                # for each old_group, split muon_params further based on size for distributed setting
                if muon_params['params']:
                    unique_sizes = {p.numel() for p in muon_params['params']}
                    device = muon_params['params'][0].device

                    for size in unique_sizes:
                        params_of_size = []
                        param_names_of_size = []

                        for p, pn in zip(muon_params['params'], muon_params['param_names']):
                            if p.numel() == size:
                                params_of_size.append(p)
                                param_names_of_size.append(pn)

                        b = torch.empty(world_size, size, dtype=torch.bfloat16, device=device)
                        muon_params_group = {
                            'params': params_of_size,
                            'param_names': param_names_of_size,
                            'update_buffer': b,
                            'update_buffer_views': [b[i] for i in range(world_size)],
                            'sharded': old_group['sharded'],
                            'max_grad_norm': old_group['max_grad_norm'],
                            'max_grad_norm_ratio': old_group['max_grad_norm_ratio'],
                            **muon_defaults,
                        }

                        param_groups.append(muon_params_group)

        del params
        super(MuonAdamW, self).__init__(
            param_groups, {}, *args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs
        )

    def zeropower_via_newtonschulz5(self, G: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X
        
        if G.size(-2) > G.size(-1):
            X = X.mT

        return X

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            optimizer_type = group['optimizer_type']

            if optimizer_type == 'muon':
                update_buffer: Tensor = group["update_buffer"]
                update_buffer_views: list[Tensor] = group["update_buffer_views"]

                # generate weight updates in distributed fashion
                params: list[Tensor] = group["params"]
                handle = None
                params_world = None

                if update_buffer.device != params[0].device:
                    update_buffer = update_buffer.to(params[0].device)
                    update_buffer_views = [b.to(params[0].device) for b in update_buffer_views]

                def update_prev():
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        p_world.mul_(1 - group["lr"] * group["weight_decay"])
                        scale = group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5
                        p_world.add_(g_world.view_as(p_world), alpha=-scale)

                for base_i in range(len(params))[::self.world_size]:
                    if base_i + self.rank < len(params):
                        p = params[base_i + self.rank]
                        g = p.grad

                        assert g is not None
                        assert not g.is_sparse

                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        if g.ndim == 4: # for the case of conv filters
                            g = g.view(len(g), -1)
                        g = self.zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                    else:
                        g = update_buffer_views[self.rank]

                    if base_i > 0:
                        update_prev()

                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    params_world = params[base_i : base_i + self.world_size]

                if handle is not None:
                    update_prev()
            elif optimizer_type == 'adamw':
                for p in group['params']:
                    if p.grad is None:
                        continue

                    # Perform stepweight decay
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                    # Perform optimization step
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    amsgrad = group['amsgrad']

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Adafactor(Optimizer):

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


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
        record_update_metrics: bool = False,
        selective_updates: bool = False,
        device: Optional[torch.device] = None,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(
            params, defaults, record_update_metrics=record_update_metrics, selective_updates=selective_updates
        )
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = device

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(
            module, FSDP
        ), "`get_post_step_metrics` expects module to be FSDP and will not work with other `distributed_strategy`."

        update_total_dot_prod = self._update_total_dot_prod
        update_total_norm = self._update_total_norm
        signed_update_total_norm = self._signed_update_total_norm
        if update_total_dot_prod is None or update_total_norm is None or signed_update_total_norm is None:
            return {}

        self._update_total_dot_prod = None
        self._update_total_norm = None
        self._signed_update_total_norm = None

        if is_distributed() and isinstance(module, FullyShardedDataParallel):
            # Reduce total dot prod and norms across all ranks.
            update_total_norm = update_total_norm**2.0
            signed_update_total_norm = signed_update_total_norm**2.0
            # Reduce all together to avoid multiple communication calls.
            all_together = torch.stack([update_total_dot_prod, update_total_norm, signed_update_total_norm])
            # Only need the final result on rank0, since that's where we log from.
            dist.reduce(
                all_together,
                0 if process_group is None else dist.get_global_rank(process_group, 0),
                group=process_group,
            )
            update_total_dot_prod, update_total_norm, signed_update_total_norm = all_together
            update_total_norm = update_total_norm**0.5
            signed_update_total_norm = signed_update_total_norm**0.5

        update_cos_sim = update_total_dot_prod / torch.max(
            update_total_norm * signed_update_total_norm,
            torch.tensor(1e-8, device=get_default_device() if self._device is None else self._device),
        )
        return {"update_cos_sim": update_cos_sim}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod: Optional[torch.Tensor] = None
        update_norms: Optional[List[torch.Tensor]] = None
        signed_update_norms: Optional[List[torch.Tensor]] = None
        if self._collecting_metrics and self._record_update_metrics:
            update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
            update_norms = []
            signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]

                # Perform step weight decay
                mask: Union[torch.Tensor, int] = grad != 0 if self._selective_updates else 1
                p.data.mul_(1 - mask * (group["lr"] * group["weight_decay"]))

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                if isinstance(mask, torch.Tensor):
                    # When mask isn't a tensor it's just a literal `1` (python int), so there's
                    # no point in calling this op.
                    update.mul_(mask)
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(1 - mask * (1 - beta2)).add_(grad, alpha=1 - beta2)

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                if (
                    update_total_dot_prod is not None
                    and update_norms is not None
                    and signed_update_norms is not None
                ):
                    update_total_dot_prod = update_total_dot_prod.to(update.device)
                    update_total_dot_prod += torch.tensordot(update, signed_update, dims=len(update.shape))
                    update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))
                    signed_update_norms.append(torch.linalg.vector_norm(signed_update, 2.0, dtype=torch.float32))

        # Compute cosine similarity between update and signed update.
        if update_total_dot_prod is not None and update_norms is not None and signed_update_norms is not None:
            device = get_default_device() if self._device is None else self._device
            self._update_total_dot_prod = update_total_dot_prod.to(device)
            self._update_total_norm = torch.linalg.vector_norm(
                torch.stack(update_norms),
                2.0,
                dtype=torch.float32,
            ).to(device)
            self._signed_update_total_norm = torch.linalg.vector_norm(
                torch.stack(signed_update_norms),
                2.0,
                dtype=torch.float32,
            ).to(device)


class AdamW(torch.optim.AdamW, Optimizer):
    def __init__(self, *args, record_update_metrics: bool = False, selective_updates: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        # Need to set these here just like in our base `Optimizer` class since our `Optimizer.__init__`
        # won't be called.
        self._record_update_metrics = record_update_metrics
        self._collecting_metrics = False
        self._selective_updates = selective_updates

        self._step_size_param_names: Optional[List[str]] = None
        self._step_size_norms: Optional[List[torch.Tensor]] = None
        self._step_size_maxs: Optional[List[torch.Tensor]] = None

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if not (self._record_update_metrics and self._collecting_metrics) and not self._selective_updates:
            return super().step(closure=closure)

        device = get_default_device()
        param_names = []
        step_size_norms = []
        step_size_maxs = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            amsgrad = group["amsgrad"]
            for name, param in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                param_names.append(name)
                grad = param.grad
                if grad is None:
                    step_size_norms.append(torch.tensor([0.0], device=device))
                    step_size_maxs.append(torch.tensor([0.0], device=device))
                    continue

                state = self.state[param]
                # init state if needed
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((), dtype=torch.float32, device=param.device)
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=torch.float32)
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]

                # Update step.
                step_t += 1

                # Perform step weight decay.
                mask: Union[torch.Tensor, int] = grad != 0 if self._selective_updates else 1
                param.mul_(1 - mask * (lr * weight_decay))

                # Decay the first and second moment running average coefficient.
                exp_avg.lerp_(grad, mask * (1 - beta1))
                exp_avg_sq.mul_(1 - mask * (1 - beta2)).addcmul_(grad, grad, value=1 - beta2)

                step = step_t.item()

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = sqrt(bias_correction2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                update = -step_size * torch.div(exp_avg, denom)
                if isinstance(mask, torch.Tensor):
                    # When mask isn't a tensor it's just a literal `1` (python int), so there's
                    # no point in calling this op.
                    update.mul_(mask)
                param.add_(update)
                step_size_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32).unsqueeze(0))
                step_size_maxs.append(update.abs().max().unsqueeze(0))

        self._step_size_param_names = param_names
        self._step_size_norms = step_size_norms
        self._step_size_maxs = step_size_maxs

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        if not (self._record_update_metrics and self._collecting_metrics):
            return {}
        else:
            device = get_default_device()
            dst_rank = 0
            if process_group is not None:
                dst_rank = dist.get_global_rank(process_group, 0)
            param_names = self._step_size_param_names
            step_size_norms = self._step_size_norms
            step_size_maxs = self._step_size_maxs
            assert param_names is not None
            assert step_size_norms is not None
            assert step_size_maxs is not None

            # Reduce metrics if needed.
            if is_distributed() and isinstance(module, FullyShardedDataParallel):
                # Reduce norms.
                all_norms = torch.cat(step_size_norms).to(device) ** 2.0
                dist.reduce(all_norms, dst_rank, op=dist.ReduceOp.SUM, group=process_group)
                step_size_norms = (all_norms ** (0.5)).squeeze(0).split(1)

                # Reduce maxs.
                all_maxs = torch.cat(step_size_maxs).to(device)
                dist.reduce(all_maxs, dst_rank, op=dist.ReduceOp.MAX, group=process_group)
                step_size_maxs = all_maxs.split(1)

            metrics = {}
            for param_name, step_size_norm, step_size_max in zip(param_names, step_size_norms, step_size_maxs):  # type: ignore[arg-type]
                metrics[f"step/{param_name}.norm"] = step_size_norm.squeeze(0)
                metrics[f"step/{param_name}.max"] = step_size_max.squeeze(0)

            self._step_size_param_names = None
            self._step_size_norms = None
            self._step_size_maxs = None
            return metrics


class Adan(Optimizer):

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


class AdEMAMix(Optimizer):

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


class AdamWScheduleFree(Optimizer):
    r"""
    Schedule-Free AdamW: adapted for OLMo from: https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py

    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.
    
    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
    """
    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 0.0025,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
        *args,
        record_update_metrics = False,
        selective_updates = False,
        **kwargs,
    ):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps,
            r=r,
            k=0,
            warmup_steps=warmup_steps,
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            foreach=foreach,
        )

        super().__init__(params, defaults, *args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(p.device), weight=1-1/beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1-beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']

            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0

            bias_correction2 = 1 - beta2 ** (k+1)
            lr = group['lr']*sched
            group['scheduled_lr'] = lr # For logging purposes

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight/weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if group['foreach'] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(*[(p, 
                                                p.grad, 
                                                self.state[p]['exp_avg_sq'], 
                                                self.state[p]['z']) 
                                                for p in active_p])

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1-beta2)
                denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)

                # Normalize grad in-place for memory efficiency
                torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr*(beta1*(1-ckp1)-1))

                # z step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p # Notation to match theory
                    grad = p.grad

                    state = self.state[p]

                    z = state['z']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    # Reuse grad buffer for memory efficiency
                    grad_normalized = grad.div_(denom)

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr*(beta1*(1-ckp1)-1))

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            group['k'] = k+1

        return loss


class Soap(Optimizer):

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


class Shampoo(Optimizer):

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


@dataclass
class Scheduler(metaclass=ABCMeta):
    # NOTE: these fields are not given default values because otherwise dataclasses complains
    # about how the scheduler subclasses are defined.
    grad_clip_warmup_steps: Optional[int]
    grad_clip_warmup_factor: Optional[float]
    warmup_min_lr: Optional[float]

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        del max_steps  # might need this in the future, but for now I just wanted to match the API of `get_lr()`.
        if initial_value is None:
            return None
        elif (
            self.grad_clip_warmup_steps is None
            or self.grad_clip_warmup_factor is None
            or step > self.grad_clip_warmup_steps
        ):
            return initial_value
        else:
            return self.grad_clip_warmup_factor * initial_value

    def get_max_grad_norm(
        self, initial_max_grad_norm: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm, step, max_steps)

    def get_max_grad_norm_ratio(
        self, initial_max_grad_norm_ratio: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm_ratio, step, max_steps)

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000) -> float:
        warmup_min_lr = self.warmup_min_lr if self.warmup_min_lr is not None else initial_lr * 0.10
        assert 0 <= warmup_min_lr < initial_lr
        return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


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
class LinearWithWarmup(Scheduler):
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
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


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


@dataclass
class BoltOnWarmupScheduler(Scheduler):
    inner: Scheduler
    warmup_start: int
    warmup_end: int

    @classmethod
    def wrap(cls, scheduler: Scheduler, warmup_start: int, warmup_end: int) -> "BoltOnWarmupScheduler":
        return cls(
            grad_clip_warmup_steps=None,
            grad_clip_warmup_factor=None,
            inner=scheduler,
            warmup_start=warmup_start,
            warmup_end=warmup_end,
            warmup_min_lr=None,
        )

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_start:
            return 0.0
        if step < self.warmup_end:
            lr_at_intercept = self.inner.get_lr(initial_lr, self.warmup_end, max_steps)
            return lr_at_intercept * (step - self.warmup_start) / (self.warmup_end - self.warmup_start)
        else:
            return self.inner.get_lr(initial_lr, step, max_steps)

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self.inner._get_max_grad_norm_coeff(initial_value, step, max_steps)


@dataclass
class ConstantScheduler(Scheduler):
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        del step, max_steps
        return initial_lr


@dataclass
class CosLinearEnvelope(Scheduler):
    "Pointwise product of cosine schedule and linear decay; useful during annealing."
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        if step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            linear_envelope = 1 - (step / max_steps)
            cosine_schedule = (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2
            return eta_min + linear_envelope * cosine_schedule


@dataclass
class ConstantWithWarmupScheduler(Scheduler):
    warmup_steps: int

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        del max_steps
        return initial_lr


PARAM_GROUP_FIELDS = ("sharded", "max_grad_norm", "max_grad_norm_ratio", "param_names")


def get_param_groups(cfg: TrainConfig, model: nn.Module) -> List[Dict[str, Any]]:
    """
    Separate parameters into weight decay and non weight decay groups.
    """
    param_groups: List[Dict[str, Any]]
    param_group_defaults = {
        "sharded": isinstance(model, FullyShardedDataParallel),
        "max_grad_norm": cfg.max_grad_norm,
        "max_grad_norm_ratio": cfg.max_grad_norm_ratio,
    }

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
                if cfg.optimizer.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Linear):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (LayerNormBase, nn.LayerNorm, RMSLayerNorm)):
                if cfg.optimizer.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Embedding):
                if cfg.optimizer.decay_embeddings:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    decay_sorted = sorted(list(decay))
    no_decay_sorted = sorted(list(no_decay))
    param_groups = []
    if len(decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in decay_sorted],
                "param_names": decay_sorted,
                **param_group_defaults,
            }
        )
    if len(no_decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in no_decay_sorted],
                "param_names": no_decay_sorted,
                "weight_decay": 0.0,
                **param_group_defaults,
            }
        )

    # Validate fields.
    for group in param_groups:
        for key in PARAM_GROUP_FIELDS:
            assert key in group

    return param_groups


def fix_optim_state_dict(optimizer: Optimizer, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure old optim state dicts are compatible with new versions.
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

    assert len(optimizer.param_groups) == len(state_dict["param_groups"])

    # Make sure:
    #  - All required fields are included in the state dict,
    #  - And that the values of those fields doesn't change from what's currently set in the optimizer,
    #    since we might have changed those fields on purpose after a restart.
    for group, sd_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for key in PARAM_GROUP_FIELDS:
            sd_group[key] = group[key]

    return state_dict


def build_optimizer(cfg: TrainConfig, model: nn.Module, world_size: int, rank: int) -> Optimizer:
    param_groups = get_param_groups(cfg, model)
    log.info(f"Constructing optimizer with {len(param_groups)} param groups")
    if cfg.optimizer.name == OptimizerType.lionw:
        return LionW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            record_update_metrics=cfg.optimizer.record_update_metrics,
            selective_updates=cfg.optimizer.selective_updates,
        )
    elif cfg.optimizer.name == OptimizerType.adamw:
        return AdamW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            record_update_metrics=cfg.optimizer.record_update_metrics,
            selective_updates=cfg.optimizer.selective_updates,
            eps=cfg.optimizer.eps,
        )
    elif cfg.optimizer.name == OptimizerType.muon:
        return MuonAdamW(
            param_groups,
            muon_lr=cfg.optimizer.muon_lr,
            muon_weight_decay=cfg.optimizer.muon_weight_decay,
            muon_momentum=cfg.optimizer.muon_momentum,
            nesterov=cfg.optimizer.nesterov,
            ns_steps=cfg.optimizer.ns_steps,
            rank=rank,
            world_size=world_size,
            adam_lr=cfg.optimizer.learning_rate,
            adam_betas=cfg.optimizer.betas,
            adam_weight_decay=cfg.optimizer.weight_decay,
            record_update_metrics=cfg.optimizer.record_update_metrics,
            selective_updates=cfg.optimizer.selective_updates,
            eps=cfg.optimizer.eps,
            vocab_size=cfg.model.embedding_size,
        )
    elif cfg.optimizer.name == OptimizerType.schedule_free_adamw:
        return AdamWScheduleFree(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
            warmup_steps=cfg.scheduler.t_warmup,
            record_update_metrics=cfg.optimizer.record_update_metrics,
            selective_updates=cfg.optimizer.selective_updates,
        )
    else:
        raise NotImplementedError


def build_scheduler(cfg: TrainConfig, sched_cfg: Optional[SchedulerConfig] = None) -> Scheduler:
    sched_cfg = sched_cfg if sched_cfg is not None else cfg.scheduler
    if sched_cfg.name == SchedulerType.cosine_with_warmup:
        return CosWithWarmup(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.linear_with_warmup:
        return LinearWithWarmup(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.inverse_sqrt_with_warmup:
        return InvSqrtWithWarmup(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.max_scheduler:
        return MaxScheduler(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            sched1=build_scheduler(cfg, replace(sched_cfg, name=SchedulerType.cosine_with_warmup)),
            sched2=build_scheduler(cfg, replace(sched_cfg, name=SchedulerType.inverse_sqrt_with_warmup)),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.constant:
        return ConstantScheduler(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.cosine_linear_envelope:
        return CosLinearEnvelope(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.constant_with_warmup:
        return ConstantWithWarmupScheduler(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_min_lr=sched_cfg.warmup_min_lr,
            warmup_steps=int(sched_cfg.t_warmup),
        )
    else:
        raise NotImplementedError
