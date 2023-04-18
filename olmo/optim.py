import warnings
from typing import Callable, Optional, Tuple, cast

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["DecoupledLionW"]


class DecoupledLionW(Optimizer):
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
