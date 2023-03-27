import math
import warnings
from typing import Callable, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from .config import OptimizerType

__all__ = ["DecoupledLionW", "build_optimizer"]


class DecoupledLionW(Optimizer):
    metric_functions = {
        "l2_norm/moment": lambda param, optim_state, step_tensor: torch.linalg.vector_norm(  # pyright: ignore
            optim_state["exp_avg"]
        ),
        "l2_norm/param": lambda param, optim_state, step_tensor: torch.linalg.vector_norm(  # pyright: ignore
            param.data
        ),
        "l2_norm/update": lambda param, optim_state, step_tensor: torch.linalg.vector_norm(  # pyright: ignore
            step_tensor
        ),
        "l2_norm/grad": lambda param, optim_state, step_tensor: torch.linalg.vector_norm(  # pyright: ignore
            param.grad
        ),
        "cosine/update_grad": lambda param, optim_state, step_tensor: F.cosine_similarity(  # pyright: ignore
            param.grad.flatten(), step_tensor.flatten(), dim=0
        ),
        "cosine/moment_grad": lambda param, optim_state, step_tensor: F.cosine_similarity(  # pyright: ignore
            param.grad.flatten(), optim_state["exp_avg"].flatten(), dim=0
        ),
    }

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        if weight_decay >= 1e-3:
            warnings.warn(
                f"You are using a high value of `weight_decay={weight_decay}` for the `DecoupledLionW` optimizer. "
                f"Are you sure you want to do this? "
                f"Your model's weights will be multiplied by {1.0 - weight_decay} on every step!",
                UserWarning,
            )

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @staticmethod
    def lionw(
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        initial_lr: float,
        wd: float,
        beta1: float,
        beta2: float,
    ) -> None:
        # step weight decay
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        # update is interpolation between gradient and momentum
        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # momentum is interpolation between gradient and itself
        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None and p.requires_grad, group["params"]):
                grad, lr, initial_lr, wd, beta1, beta2, state = (  # type: ignore
                    cast(torch.Tensor, p.grad),
                    cast(float, group["lr"]),
                    cast(float, group["initial_lr"]),
                    cast(float, group["weight_decay"]),
                    *cast(Tuple[float, float], group["betas"]),
                    cast(dict, self.state[p]),
                )

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                self.lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2)

        return loss

    def dist_reduce_metrics(self, optimizer_metrics):
        from composer.utils import dist

        for metric in optimizer_metrics:
            if metric.startswith("l2_norm"):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation="SUM")

                optimizer_metrics[metric] = math.sqrt(reduced)
            elif metric.startswith("cosine"):
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation="SUM")

                _, vectors, layer = tuple(metric.split("/"))

                A, B = tuple(vectors.split("_"))

                A_reduced_norm = optimizer_metrics[f"l2_norm/{A}/{layer}"]
                B_reduced_norm = optimizer_metrics[f"l2_norm/{B}/{layer}"]
                optimizer_metrics[metric] = reduced / (A_reduced_norm * B_reduced_norm)
            else:
                reduced = optimizer_metrics[metric]
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation="SUM")
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics):
        """Preprocess metrics to reduce across ranks correctly."""
        # Sort L2 norms first so they are squared before other metrics, which depend on squared values
        metrics = optimizer_metrics.keys()
        metrics = sorted(metrics, key=lambda metric: 0 if "l2_norm" in metric else 1)
        for metric in metrics:
            if metric.startswith("l2_norm"):
                # L2 norms need to be squared, before they are reduced via summation
                optimizer_metrics[metric] = optimizer_metrics[metric] ** 2
            elif metric.startswith("cosine"):
                _, vectors, layer = tuple(metric.split("/"))

                A, B = tuple(vectors.split("_"))

                # L2 norm would've been squared in previous branch
                A_rank_subset_norm = math.sqrt(optimizer_metrics[f"l2_norm/{A}/{layer}"])
                B_rank_subset_norm = math.sqrt(optimizer_metrics[f"l2_norm/{B}/{layer}"])

                optimizer_metrics[metric] *= A_rank_subset_norm * B_rank_subset_norm

        return optimizer_metrics

    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        lr = self.param_groups[0]["lr"]
        weight_decay = self.param_groups[0]["weight_decay"]
        initial_lr = self.param_groups[0]["initial_lr"]

        beta1, _ = self.param_groups[0]["betas"]
        if param in self.state:
            param_optim_state = self.state[param]
            step_tensor = param_optim_state["exp_avg"].clone().lerp_(param.grad, 1 - beta1).sign_().mul_(lr)
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f"{metric}/{name}"] = self.metric_functions[metric](
                    param, param_optim_state, step_tensor
                )

        return optimizer_metrics


def build_optimizer(
    model,
    name: OptimizerType = OptimizerType.decoupled_lionw,
    learning_rate: Optional[float] = None,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Get a suitable optimizer for training/fine-tuning.

    :param learning_rate: The learning rate. If not specified, a default learning
        rate will calculated according to the equation from the Scaling Laws paper
        `0.003239 - 0.0001395 * math.log(N)`,
        where `N` is the number of trainable parameters excluding embeddings.
    :param weight_decay: The weight decay coefficient. This does not apply to
        biases and layernorm/embedding weights, which will have a weight decay
        coefficient of 0.
    :param kwargs: Other keyword arguments passed to the optimizer.
    """
    # Separate out all parameters to those that will and won't experience regularizing weight decay.
    decay = set()
    no_decay = set()
    all_params = {}
    num_trainable_non_embedding_weights = 0
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
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.LayerNorm, nn.Embedding)):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

            if fpn not in {"transformer.wte.weight", "transformer.wpe.weight"}:
                num_trainable_non_embedding_weights += p.numel()

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    optim_groups = [
        {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if learning_rate is None:
        learning_rate = 0.003239 - 0.0001395 * math.log(num_trainable_non_embedding_weights)

    if name == OptimizerType.decoupled_lionw:
        return DecoupledLionW(optim_groups, lr=learning_rate, betas=betas)
    elif name == OptimizerType.decoupled_adamw:
        from composer.optim import DecoupledAdamW

        return DecoupledAdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    elif name == OptimizerType.adamw:
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise NotImplementedError(f"Not sure how to build optimizer '{name}'")
