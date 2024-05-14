import os
import pytest
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from olmo import OLMo
from olmo.train import Trainer
from olmo.config import TrainConfig
from olmo.model import LayerNormBase
from olmo.data import build_train_dataloader
from olmo.optim import build_optimizer, build_scheduler
from olmo.torch_util import get_world_size, get_local_rank, get_default_device

from packaging import version
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support, _device_has_foreach_support


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
    cfg.data.paths = ["test_fixtures/c4-sample.01.json.gz", "test_fixtures/c4-sample.02.json.gz", "test_fixtures/c4-sample.03.json.gz"]
    cfg.model.vocab_size = 2**16  # some tokens in sample files are upto 65k
    cfg.model.embedding_size = cfg.model.vocab_size # this gives an error without this
    cfg.model.weight_tying = False
    cfg.model.rope = True

    cfg.optimizer.name = "adamw"
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.name = "cosine_with_warmup"
    cfg.scheduler.units = "steps"
    cfg.scheduler.t_warmup = 100
    cfg.scheduler.t_max = 1000
    cfg.scheduler.alpha_f = 0.  # our custom test scheduler decays to 0
    cfg.max_grad_norm = max_norm
    cfg.seed = 6198

    return cfg


def _no_grad(func):
    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


@_no_grad
def clip_grad_norm_(
        parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(grads) == 0:
        return torch.tensor(0.)

    first_device = grads[0].device
    grouped_grads = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms = []
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        if (
            (foreach is None and _has_foreach_support(device_grads, device))
            or (foreach and _device_has_foreach_support(device))
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    return total_norm


def _naive_train_loop(cfg, model, optimizer, scheduler, data_loader, max_iterations, max_norm=1.0, clip_grad=False):
    len_dataloader = 3
    max_epochs = max_iterations // len_dataloader + 1
    norm_vector = []

    for epoch in range(max_epochs):
        for idx, batch in enumerate(data_loader):
            step_count = epoch * len_dataloader + idx
            optimizer.zero_grad()

            logits = model(batch['input_ids'].to('cuda')).logits

            # compute loss
            loss = _lm_loss(logits, batch['input_ids'].to('cuda').clone())
            loss.backward()

            if clip_grad:
                # norm_vector.append(nn.utils.clip_grad_norm_(model.parameters(), max_norm))
                norm_vector.append(clip_grad_norm_(model.parameters(), max_norm))
            else:
                norm_vector.append(optimizer.clip_grads_and_collect_metrics(step_count)["total_grad_norm"])

            # apply scheduler
            for group in optimizer.param_groups:
                group["lr"] = scheduler.get_lr(cfg.optimizer.learning_rate, step_count, cfg.scheduler.t_max)
                group["max_grad_norm"] = scheduler.get_max_grad_norm(cfg.max_grad_norm, step_count, cfg.scheduler.t_max)
                group["max_grad_norm_ratio"] = scheduler.get_max_grad_norm(cfg.max_grad_norm_ratio, step_count, cfg.scheduler.t_max)

            optimizer.step()

            if step_count % 100 == 0:
                print('Step: {:4d}, Loss: {:.4f}'.format(step_count, loss))

            if step_count == max_iterations:
                break

    return norm_vector


def _run_olmo_grad_norm_againt_torch_grad_norm(
        rank: int,
        world_size: int,
        max_iterations: int,
        max_norm: float,
    ):
    # step 2: run 5 batches on GPU using no_shard fsdp strategy with olmo trainer and compare clipping
    # step 3: run this for other fsdp strategies

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

    print('pytorch optim, pytorch grad_clipping, no model wrapping...')

    # run on CPU
    model = OLMo(cfg.model).to('cuda')
    torch_optimizer = _init_torch_optim(cfg, model)
    scheduler = build_scheduler(cfg)
    data_loader = build_train_dataloader(cfg)

    torch_optim_norms = _naive_train_loop(cfg, model, torch_optimizer, scheduler, data_loader, max_iterations, max_norm, clip_grad=True)

    del model
    del torch_optimizer
    del scheduler
    del data_loader

    # use same model, data, optimizer, fsdp_model and send to trainer and compare gradient clip

    # olmo optimizer
    model = OLMo(cfg.model).to('cuda')
    olmo_optimizer = build_optimizer(cfg, model)
    # torch_optimizer = _init_torch_optim(cfg, model)
    data_loader = build_train_dataloader(cfg)
    scheduler = build_scheduler(cfg)

    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=get_default_device())

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None

    # build fsdp config
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        auto_wrap_policy=None,
        use_orig_params=True,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        param_init_fn=param_init_fn,
    )

    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if param_init_fn is not None:
        model.reset_parameters()

    print('olmo optim, olmo grad_clipping, fsdp no_shard wrapping...')
    # print('torch optim, torch grad_clipping, fsdp no_shard wrapping...')

    olmo_optim_norms = _naive_train_loop(cfg, fsdp_model, olmo_optimizer, scheduler, data_loader, max_iterations)
    # olmo_optim_norms = _naive_train_loop(cfg, model, torch_optimizer, scheduler, data_loader, max_iterations, max_norm, clip_grad=True)

    # print('###############################################')
    # print(torch_optim_norms)
    # print(olmo_optim_norms)
    # print('###############################################')

    # Shut down world pg
    dist.destroy_process_group()

    # TODO: remove
    assert False


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires 1 CUDA device")
@pytest.mark.parametrize("max_iterations, max_norm", [pytest.param(1000, 1.0)])
def test_local_sharded_checkpointer(max_iterations, max_norm):
    world_size = 1
    # TODO: must run fsdp clipping with world_size > 1
    # world_size = torch.cuda.device_count()
    mp.spawn(
        _run_olmo_grad_norm_againt_torch_grad_norm,
        args=(world_size, max_iterations, max_norm),
        nprocs=world_size,
        join=True,
    )
