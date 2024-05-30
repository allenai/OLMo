import functools
import os
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from olmo import OLMo
from olmo.config import TrainConfig
from olmo.data import build_train_dataloader
from olmo.model import LayerNormBase
from olmo.optim import build_optimizer, build_scheduler
from olmo.torch_util import get_world_size, seed_all


def _lm_loss(logits, labels):
    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    return F.cross_entropy(logits, labels)


def _init_torch_optim(cfg, model):
    """
    This matches the decay/no-decay param group split without the gradient clipping.
    We clip the gradients separately for test.
    """
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
    optim_groups = [
        {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": cfg.optimizer.weight_decay},
        {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
    )

    return optimizer


def _patch_config(cfg, max_norm):
    # patch config
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    cfg.data.paths = [
        "test_fixtures/c4-sample.01.json.gz",
        "test_fixtures/c4-sample.02.json.gz",
        "test_fixtures/c4-sample.03.json.gz",
    ]
    cfg.model.vocab_size = 2**16  # some tokens in sample files are upto 65k
    cfg.model.embedding_size = cfg.model.vocab_size  # this gives an error without this
    cfg.model.weight_tying = False
    cfg.model.rope = True

    cfg.optimizer.name = "adamw"
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.eps = 1e-8
    cfg.scheduler.name = "constant"
    cfg.scheduler.units = "steps"
    cfg.scheduler.t_warmup = 100
    cfg.scheduler.t_max = 1000
    cfg.scheduler.alpha_f = 0.0  # our custom test scheduler decays to 0
    cfg.max_grad_norm = max_norm
    cfg.seed = 6198

    cfg.model.attention_dropout = 0.0
    cfg.model.residual_dropout = 0.0
    cfg.model.embedding_dropout = 0.0

    return cfg


def _apply_scheduler(cfg, step_count, scheduler, optimizer):
    """
    Apply scheduler according to OLMo style.
    """
    for group in optimizer.param_groups:
        group["lr"] = scheduler.get_lr(cfg.optimizer.learning_rate, step_count, cfg.scheduler.t_max)
        group["max_grad_norm"] = scheduler.get_max_grad_norm(cfg.max_grad_norm, step_count, cfg.scheduler.t_max)
        group["max_grad_norm_ratio"] = scheduler.get_max_grad_norm(
            cfg.max_grad_norm_ratio, step_count, cfg.scheduler.t_max
        )

    return optimizer.param_groups[0]["lr"]


def get_state_with_grads(model):
    state_dict = {}

    for name, param in model.named_parameters():
        state_dict[name] = deepcopy(param)
        state_dict[name].grad = deepcopy(param.grad)

    return state_dict


def _naive_train_loop(
    cfg,
    model_a,
    model_b,
    optimizer_a,
    optimizer_b,
    scheduler_a,
    scheduler_b,
    data_loader,
    max_iterations,
    max_norm=1.0,
    device="cpu",
):
    """
    Naive torch training loop.
    """
    len_dataloader = 3
    max_epochs = max_iterations // len_dataloader + 1

    model_a_init_state = get_state_with_grads(model_a)
    model_b_init_state = get_state_with_grads(model_b)

    total_param_diff = 0
    for name in model_a_init_state.keys():
        total_param_diff += (model_a_init_state[name] - model_b_init_state[name]).abs().sum()

    assert total_param_diff == 0.0, "models are not initialized correctly"

    for epoch in range(max_epochs):
        for idx, batch in enumerate(data_loader):
            step_count = epoch * len_dataloader + idx

            optimizer_a.zero_grad()
            seed_all(step_count)

            logits_a = model_a(batch["input_ids"].to(device)).logits
            loss_a = _lm_loss(logits_a, batch["input_ids"].to(device).clone())

            loss_a.backward()
            torch_grad_norm = clip_grad_norm_(model_a.parameters(), max_norm)

            _apply_scheduler(cfg, step_count, scheduler_a, optimizer_a)
            optimizer_a.step()

            ####################################################################

            optimizer_b.zero_grad()
            seed_all(step_count)

            logits_b = model_b(batch["input_ids"].to(device)).logits
            loss_b = _lm_loss(logits_b, batch["input_ids"].to(device).clone())

            loss_b.backward()
            olmo_grad_norm = optimizer_b.clip_grads_and_collect_metrics(step_count, device=torch.device(device))[
                "total_grad_norm"
            ]

            _apply_scheduler(cfg, step_count, scheduler_b, optimizer_b)
            optimizer_b.step()

            total_param_diff = 0
            total_grad_diff = 0

            model_a_state = get_state_with_grads(model_a)
            model_b_state = get_state_with_grads(model_b)

            for name in model_a_state.keys():
                param_diff = (model_a_state[name] - model_b_state[name]).abs().sum()
                grad_diff = (model_a_state[name].grad - model_b_state[name].grad).abs().sum()

                total_param_diff += param_diff
                total_grad_diff += grad_diff

            # params set by observing grads for the two cases on a cpu run
            assert total_grad_diff < 1e-4, "model gradients diverged during optimization"
            assert total_param_diff < 1e-2, "model parameters diverged during optimization"
            assert (
                torch.abs(torch_grad_norm - olmo_grad_norm) < 1e-6
            ), "grad norms computed by torch and OLMo codebase are different"

            if step_count == max_iterations:
                break


def _run_olmo_optim_againt_torch_optim(
    rank: int,
    world_size: int,
    max_iterations: int,
    max_norm: float,
    device: str,
):
    # minimal distributed env setup
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # Initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cfg = TrainConfig.load("test_fixtures/train_tiny.yaml")
    cfg = _patch_config(cfg, max_norm)

    seed_all(cfg.seed)

    model_a = OLMo(cfg.model).to(device)
    torch_optimizer = _init_torch_optim(cfg, model_a)
    scheduler_a = build_scheduler(cfg)
    data_loader = build_train_dataloader(cfg)

    # olmo optimizer
    model_b = deepcopy(model_a)
    olmo_optimizer = build_optimizer(cfg, model_b)
    scheduler_b = build_scheduler(cfg)

    _naive_train_loop(
        cfg=cfg,
        model_a=model_a,
        model_b=model_b,
        optimizer_a=torch_optimizer,
        optimizer_b=olmo_optimizer,
        scheduler_a=scheduler_a,
        scheduler_b=scheduler_b,
        data_loader=data_loader,
        max_iterations=max_iterations,
        max_norm=max_norm,
        device=device,
    )

    # Shut down world pg
    dist.destroy_process_group()


@pytest.mark.parametrize("max_iterations, max_norm, device", [pytest.param(10, 1.0, "cpu")])
def test_olmo_optimizer_and_clipping_cpu(max_iterations, max_norm, device):
    world_size = 1
    mp.spawn(
        _run_olmo_optim_againt_torch_optim,
        args=(world_size, max_iterations, max_norm, device),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires 1 CUDA device")
@pytest.mark.parametrize("max_iterations, max_norm, device", [pytest.param(10, 1.0, "cuda")])
def test_olmo_optimizer_and_clipping_gpu(max_iterations, max_norm, device):
    world_size = 1
    # world_size = torch.cuda.device_count()
    mp.spawn(
        _run_olmo_optim_againt_torch_optim,
        args=(world_size, max_iterations, max_norm, device),
        nprocs=world_size,
        join=True,
    )
