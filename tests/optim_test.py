import pytest

from olmo.optim import BoltOnWarmupScheduler, LinearWithWarmup


def test_linear_with_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None, grad_clip_warmup_factor=None, warmup_steps=2000, warmup_min_lr=None
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 2000, max_steps) == 1.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) > scheduler.get_lr(initial_lr, 5_000, max_steps)


def test_bolt_on_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 11_000
    alpha_f = 0.1
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_steps=1000,
        alpha_f=alpha_f,
        warmup_min_lr=None,
    )
    scheduler2 = BoltOnWarmupScheduler.wrap(scheduler, 5000, 6000)
    assert scheduler.get_lr(initial_lr, 100, max_steps) > 0.0
    assert scheduler2.get_lr(initial_lr, 100, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5000, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5500, max_steps) == pytest.approx(0.25 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 6000, max_steps) == pytest.approx(0.5 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 7000, max_steps) == scheduler.get_lr(initial_lr, 7000, max_steps)
