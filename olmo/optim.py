import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.optimizer import Optimizer as OptimizerBase

from . import LayerNormBase
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


log = logging.getLogger(__name__)


class Optimizer(OptimizerBase):
    def __init__(self, *args, record_update_metrics: bool = False, selective_updates: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._record_update_metrics = record_update_metrics
        self._collecting_metrics = False
        self._selective_updates = selective_updates

    def _clean_param_name(self, name: str) -> str:
        name = name.replace("_fsdp_wrapped_module.", "")
        name = name.replace("module.transformer.", "")
        name = name.replace(".weight", "")

        if 'blocks' in name:
            name_split = name.split('.')
            name = name_split[0] + '_' + name_split[1] + '/' + name_split[2]

        return name

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

        adding a new metric category, exclude from printing to console: log_metrics_to_console in train.py
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
        per_param_param_names: List[str] = []
        per_param_param_tensors: List[torch.Tensor] = []
        per_param_param_norm: List[torch.Tensor] = []

        per_param_grad_names: List[str] = []
        per_param_grad_tensors: List[torch.Tensor] = []
        per_param_grad_norm: List[torch.Tensor] = []

        per_param_grad_state_1_names: List[str] = []
        per_param_grad_state_1_tensors: List[torch.Tensor] = []
        per_param_grad_state_1_norm: List[torch.Tensor] = []

        per_param_grad_state_2_names: List[str] = []
        per_param_grad_state_2_tensors: List[torch.Tensor] = []
        per_param_grad_state_2_norm: List[torch.Tensor] = []

        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)

        #######################################################################
        # part 1: collect metrics locally
        #######################################################################
        def get_norm(x, norm_list):
            # grad or state tensors could be none for params that have their shards completely on other ranks.
            if x is not None and x.numel() > 0:
                norm_list.extend([torch.linalg.vector_norm(x, 2.0, dtype=torch.float32).unsqueeze(0)])
            else:
                norm_list.extend([torch.tensor([0.0], device=device, dtype=torch.float32)])

        for group in self.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                # Always need to collect the norm of gradients for clipping, even if we're not collecting
                # other metrics.
                per_param_grad_names.extend([f"grad/{name}"])
                per_param_grad_tensors.extend([p.grad])
                get_norm(p.grad, per_param_grad_norm)

                if collect_param_metrics:
                    per_param_param_names.extend([f"param/{name}"])
                    per_param_param_tensors.extend([p])
                    get_norm(p, per_param_param_norm)

                    state = self.get_state_for_param(p)
                    sorted_state_keys = sorted([k for k in state.keys()])

                    per_param_grad_state_1_names.extend([f"exp_avg/{name}"])
                    per_param_grad_state_1_tensors.extend([state['exp_avg']])
                    get_norm(state['exp_avg'], per_param_grad_state_1_norm)

                    per_param_grad_state_2_names.extend([f"exp_avg_sq/{name}"])
                    per_param_grad_state_2_tensors.extend([state['exp_avg_sq']])
                    get_norm(state['exp_avg_sq'], per_param_grad_state_2_norm)

        assert (
            len(per_param_grad_names)
            == len(per_param_grad_tensors)
            == len(per_param_grad_norm)
            == len(per_param_param_names)
            == len(per_param_param_tensors)
            == len(per_param_param_norm)
            == len(per_param_grad_state_1_names)
            == len(per_param_grad_state_1_tensors)
            == len(per_param_grad_state_1_norm)
            == len(per_param_grad_state_2_names)
            == len(per_param_grad_state_2_tensors)
            == len(per_param_grad_state_2_norm)
        )

        #######################################################################
        # part 2: reduce metrics over ranks
        # NOTE (ananya) for DDP, this step should be unnecessary for logging params, grads and optimizer states
        # as gradients should have synchornized across ranks by the time ``clip_grads_and_collect_metrics`` is called.
        # loss.backward() is called in self.train_batch(batch)
        # NOTE (ananya) doesn't work with FSDP right now
        #######################################################################
        all_metrics: Dict[str, torch.Tensor] = {}

        total_grad_norm: torch.Tensor
        total_grad_norm = (torch.cat(per_param_grad_norm) ** 2.0).sum() ** 0.5
        all_metrics["total_grad_norm"] = total_grad_norm

        if collect_param_metrics:
            total_param_norm: torch.Tensor
            total_param_norm = (torch.cat(per_param_param_norm) ** 2.0).sum() ** 0.5
            all_metrics["total_param_norm"] = total_param_norm

            for metric_name, metric in zip(per_param_param_names, per_param_param_norm):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(per_param_grad_state_1_names, per_param_grad_state_1_norm):
                all_metrics[metric_name] = metric.squeeze(0)
            for metric_name, metric in zip(per_param_grad_state_2_names, per_param_grad_state_2_norm):
                all_metrics[metric_name] = metric.squeeze(0)

        for metric_name, metric in zip(per_param_grad_names, per_param_grad_norm):
            all_metrics[metric_name] = metric.squeeze(0)

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
            grad_norm = all_metrics.get(f"grad/{name}")
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
    
    def get_angle_between_vecs(self, mat1, mat2) -> torch.Tensor:
        if len(mat1.shape) > 2:
            raise NotImplementedError("computing angle between update and gradient not implemented for more than 2 dimensions!")

        dot_products = torch.sum(mat1 * mat2, dim=-1)

        norms_matrix1 = torch.norm(mat1, dim=-1)
        norms_matrix2 = torch.norm(mat2, dim=-1)

        cosine_angles = dot_products / (norms_matrix1 * norms_matrix2)
        cosine_angles = torch.clamp(cosine_angles, -1.0, 1.0)

        angles_radians = torch.acos(cosine_angles)

        return torch.rad2deg(angles_radians)


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


class Muon(Optimizer):
    """
        Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads,
        and scalar or vector parameters should be optimized using AdamW instead. Muon provides an internal AdamW
        for this so you don't have to use an extra optimizer.
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
        # NOTE (ananya) skipping the flag selective_updates here
        self._selective_updates = selective_updates

        self._update_names: Optional[List[str]] = None
        self._update_tensors: Optional[List[torch.Tensor]] = None
        self._update_norms: Optional[List[torch.Tensor]] = None
        self._update_grad_angles: Optional[List[torch.Tensor]] = None

    @torch.no_grad()
    def step(self, closure=None) -> None:
        # TODO: log when record_update_metrics is true, and collecting metrics flag is set
        # TODO: record the angle between gradients and update for each layer
        # TODO: compute the Hessian and its eigenvalues
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._record_update_metrics and self._collecting_metrics:
            self._update_names = []
            self._update_tensors = []
            self._update_norms = []
            self._update_grad_angles = []

        for group in self.param_groups:
            for name, p in zip(group["param_names"], group["params"]):
                if self._record_update_metrics and self._collecting_metrics:
                    name = self._clean_param_name(name)
                    self._update_names.extend([f"update/{name}"])

                if p.grad is None:
                    if self._record_update_metrics and self._collecting_metrics:
                        # add update tensors and norm here
                        self._update_tensors.extend([None])
                        self._update_norms.extend([torch.tensor([0.0], device=p.device, dtype=torch.float32)])

                        # [out, in] dimension, each layer has out dim number of neurons
                        self._update_grad_angles.extend([None])

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
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # modified addcdiv_ of OG code to get stats about update
                update = -step_size * torch.div(exp_avg, denom)
                p.add_(update)

                if self._record_update_metrics and self._collecting_metrics:
                    # add update tensors and norm here
                    self._update_tensors.extend([update])

                    if torch.isnan(update).any():
                        _update_norm = torch.linalg.vector_norm(update[~torch.isnan(update)], 2.0, dtype=torch.float32).unsqueeze(0)
                    else:
                        _update_norm = torch.linalg.vector_norm(update, 2.0, dtype=torch.float32).unsqueeze(0)
                    self._update_norms.extend([_update_norm])

                    # [out, in] dimension, each layer has out dim number of neurons
                    self._update_grad_angles.extend([self.get_angle_between_vecs(p.grad, update)])

        if self._update_names is not None:
            assert self._update_tensors is not None
            assert self._update_norms is not None
            assert self._update_grad_angles is not None

            assert (
                len(self._update_names)
                == len(self._update_tensors)
                == len(self._update_norms)
                == len(self._update_grad_angles)
            )

        return loss

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        """
            adding a new metric category, exclude from printing to console: log_metrics_to_console in train.py
        """
        metrics = {}
        assert self._update_names is not None
        assert self._update_norms is not None
        assert self._update_grad_angles is not None

        for update_name, update_norm, update_grad_angle in zip(self._update_names, self._update_norms, self._update_grad_angles):
            metrics[f"update_norm/{update_name.replace('update/', '')}"] = update_norm

            # replace all nan degrees between update/grad for wte by 400
            metrics[f"grad_update_angle/{update_name.replace('update/', '')}"] = torch.nan_to_num(update_grad_angle, nan=400)

        # TODO: add a line plot
        # TODO: plot grad/param tensors, beta1/beta2 are not normed right? they are scalers per param group?
        # TODO: plot activations
        # TODO: plot top-k eigenvalues
        self._update_names = None
        self._update_norms = None
        self._update_tensors = None
        self._update_grad_angles = None

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


class ScheduleFreeAdamW(Optimizer):

    def __init__(self, *args, record_update_metrics = False, selective_updates = False, **kwargs):
        super().__init__(*args, record_update_metrics=record_update_metrics, selective_updates=selective_updates, **kwargs)

    def get_post_step_metrics(self, module, process_group = None):
        return super().get_post_step_metrics(module, process_group)

    def get_state_for_param(self, param):
        return super().get_state_for_param(param)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        pass


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
            elif pn.endswith("weight") and isinstance(m, (LayerNormBase, nn.LayerNorm)):
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


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> Optimizer:
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
