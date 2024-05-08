import pytest

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from olmo import OLMo
from olmo.train import Trainer
from olmo.config import TrainConfig
from olmo.model import LayerNormBase
from olmo.optim import build_optimizer
from olmo.data import build_train_dataloader
from olmo.torch_util import get_world_size, get_local_rank, get_default_device

from packaging import version
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


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


def _patch_config(cfg):
    # patch config
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    cfg.data.paths = ["test_fixtures/c4-sample.01.json.gz", "test_fixtures/c4-sample.02.json.gz", "test_fixtures/c4-sample.03.json.gz"]
    cfg.model.vocab_size = 2**16  # some tokens in sample files are upto 65k
    cfg.model.embedding_size = cfg.model.vocab_size # this gives an error without this
    cfg.model.weight_tying = False
    cfg.model.rope = True

    cfg.optimizer.learning_rate = 1e-4
    cfg.scheduler.t_warmup = 100
    cfg.max_grad_norm = 1.
    cfg.seed = 6198

    cfg.fsdp_precison = ''

    return cfg


def _naive_train_loop(model, optimizer, data_loader, max_iterations, max_norm=1.0):
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

            # clip grads
            norm_vector.append(nn.utils.clip_grad_norm_(model.parameters(), max_norm))

            optimizer.step()

            print('Step: {:4d}, Loss: {:.4f}'.format(step_count, loss))

            if step_count == max_iterations:
                break

    return norm_vector


def _fsdp_no_shard_train_loop(model, optimizer, data_loader, max_iterations):
    """
    max_norm comes from cfg
    """
    norm_vector = []
    step_count = 0
    # norm_vector.append(optimizer.clip_grads_and_collect_metrics(step_count)["total_grad_norm"])
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
    cfg = _patch_config(cfg)

    # run on CPU
    model = OLMo(cfg.model).to('cuda')
    torch_optimizer = _init_torch_optim(cfg, model)
    data_loader = build_train_dataloader(cfg)

    torch_optim_norms = _naive_train_loop(model, torch_optimizer, data_loader, max_iterations, max_norm)

    del model
    del torch_optimizer
    del data_loader

    print(torch_optim_norms)
    print('################################')
    assert False
    # use same model, data, optimizer, fsdp_model and send to trainer and compare gradient clip

    # olmo optimizer
    model = OLMo(cfg.model).to('cuda')
    olmo_optimizer = build_optimizer(cfg, model)
    data_loader = build_train_dataloader(cfg)

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

    olmo_optim_norms = _fsdp_no_shard_train_loop(fsdp_model, olmo_optimizer, data_loader, max_iterations)

    # DDP
    # GPU0: 1, 3, 5
    # GPU1: 2, 4, 6
    ddp_model.forward(batch['input_ids'])

    # FSDP (GPU0: model/2 GPU1: model/2)
    # reduce_scatter
    # GPU0: 1, 3, 5
    # GPU1: 2, 4, 6
    fsdp_model.forward(batch['input_ids'])

    # print('###############################################')
    # print(torch_optim_norms)
    # print(olmo_optim_norms)
    # print('###############################################')

    # Shut down world pg
    dist.destroy_process_group()

    # TODO: remove
    assert False


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires 2 or more CUDA devices")
@pytest.mark.parametrize("max_iterations, max_norm", [pytest.param(5, 1.0)])
def test_local_sharded_checkpointer(max_iterations, max_norm):
    world_size = torch.cuda.device_count()
    mp.spawn(
        _run_olmo_grad_norm_againt_torch_grad_norm,
        args=(world_size, max_iterations, max_norm),
        nprocs=world_size,
        join=True,
    )
