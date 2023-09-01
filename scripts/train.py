"""Run this script with 'torchrun'."""

import logging
import math
import os
import sys
from typing import Dict, Any, List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

from olmo.config import TrainConfig
from olmo.exceptions import OlmoCliError
from olmo.model import Olmo
from olmo.optim import build_optimizer, build_scheduler
from olmo.util import (
    barrier,
    clean_opt,
    get_local_rank,
    get_world_size,
    log_extra_field,
    peak_gpu_memory,
    prepare_cli_environment,
    seed_all, move_to_device,
)

log = logging.getLogger("train")


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "train-llm")
    log_extra_field("run_name", cfg.run_name)

    # Initialize process group and set device.
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    barrier()

    # Initialize the model.
    log.info("Building model...")
    olmo_model = Olmo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")

    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=MixedPrecision(  # equivalent to MosaicML's "PURE"
            param_dtype=cfg.autocast_precision,
            reduce_dtype=cfg.autocast_precision,
            buffer_dtype=cfg.autocast_precision,
        ),
        auto_wrap_policy=olmo_model.fsdp_wrap_fn,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
    )

    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")

    assert not cfg.activation_checkpointing

    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    scheduler = build_scheduler(cfg)

    fsdp_model.train()

    batch = {
            "input_ids": torch.tensor([[  570, 12287,    13,  3395,   625,  2590,    15,   329, 30942,   273,
                                          5448, 22007,   949,   253,  7423,    13,   285,   187,  6972, 20625,
                                          32372,   949,   253,  9129,  2419,    15,   380, 36555,   327,   253,
                                          2829,   187,    71,   663, 40617,    15,  1244,  1335,   253,   767,
                                          2206, 15368,    14,  9458,    13, 39712,   272,  5412,  3564,    13,
                                          347,   187,   783, 39709, 44210,  4817,   689,   253,  7887,    13,
                                          285,   323,   253,  1390,   673,   187,  8732, 12469,  2205,  2210,
                                          1728,   281,  4600, 17421,   521,  5098,  1128, 45217,  1417,  1024,
                                          273,   987,   390,   187,  7041,    13,   390,  5958,   390,  2614,
                                          15,   187,   187,    11,   475,   475,   475,   475,   187,   187],
                                       [ 1439,  1469,   281,  1390,   562,   436,  2137,  2391,   309,  1849,
                                         644,  2820,   281,  2028,   368,   849,  1199,   187,  2520,   294,
                                         1763,  1596,   318,   273,  4266,   275,   368,   556,  5486,   275,
                                         253,  1390,  1643,  1107,  1051,   187,  1915,  8140, 19605,   359,
                                         403,  1051,  1095,  8140, 12401,    15,  7088,    14,  1615,    13,
                                         11761,  5006,    13,   285,  2656,   187,  1257,   342,   368,    15,
                                         4392,  8506,   947,   399, 28077,    58,    15,   535, 50268, 46928,
                                         187,   187,  3172,    35, 18004,  2637,  4915,   427,  9981,   187,
                                         187,  8096,   590,  4395,  3579,   327,   253, 12595,  1919,   344,
                                         1119,   247, 32217,   762,   271,  5637,   187,  3243,    15,   754],
                                       [  352,  4895,   281,   697, 15178,    13,   533,  3517,  2210,   896,
                                          187,  3113,  2067,  2571,    15,  2053,    13,  7824,  3790,   253,
                                          28335,    81, 33787,    13,  6730,   352,   187,  6672,  2920,    13,
                                          347,   858,   253,   806,    15,  9067,   597,   512,  2427,    28,
                                          3517,   597,  4895,   342,   187, 23350,   625,   273,   616, 24835,
                                          13,   840,   597,  4447,   247,  1048,  5084,    13,  1077,   187,
                                          3022,   247, 20762,    13,   285, 22944,   779,   281,   253,  5024,
                                          273,   253,  1854,    15,   380, 15178,  1146,   275,   187,   783,
                                          17514,    14,  3026,    84,   597,   574,   281,  3785,   779,   689,
                                          253, 28390,  8704,   253,   187,    72,  6702,    13,   533,  2378],
                                       [ 1659,   835,   352,   369,   187,  7053,  2326,    15,   380,  3884,
                                         273,   253,  1854,   369,  3164,   432,   411,    15,   407,   322,
                                         15,   281,   187,    38,    15,   407,   427,    15,   380, 31710,
                                         369,  9090,  1469,   432,   253, 19969,    13,   697,  1854,   187,
                                         11849,   271,  6907,   342,   253, 26961,  7844,   273,   670,  3925,
                                         3272,    15,   187,   187,    89, 12211,    15, 17842,  4314,   285,
                                         3578,  2909,   846,   253, 31852,   273,   253,   187,  3899, 22683,
                                         13,   627,   369,   247, 11216,   285,  5536,  1304,    13, 11704,
                                         407,   247,  1077,   187,  6209,   917, 21152,    28,   352,   858,
                                         417,  1199, 28788,  2057, 24511,   390,   253,  1304,   187,  1171],
                                       [ 1608,   482,   327,   368,  1476,   187,   187,    52,   798,   272,
                                         11306,   347,   344, 16535,  1066,   344, 19336,   521,  2159, 14228,
                                         1411,   247,   187, 10228,    13,   285,  2206,  1066,   327,   521,
                                         25296,  2822,   281, 10867, 11304,    15,   187,   187,     3,  4497,
                                         937,   344, 30807,   275, 13775,    13,   346, 30875,   352,   310,
                                         417,  7154,   449,   187,   187,     3,  3650,  7154,     2,  6049,
                                         48570,   369,  4704,   281,  2740,   619,  3802,   483,   323,   479,
                                         15,  2652,  1137,  1476,   187,   187,     3,  1989,   752,   858,
                                         368,   439,   710,   323,  1476,   187,   187,     3,  1276,   323,
                                         315,     3, 13892, 14050, 10867, 11304,    15,   346,    42,   369],
                                       [ 5313,   326,  2168,   187, 20774,   447,   352,    15,  1916,   346,
                                         12467,     3,   253,  4450,   310,   281,   452,   253,  7437,   594,
                                         4845,   187,  3529,   247,  1270,  1142,   273,   253, 19325,   403,
                                         26814,    15,   187,   187,    36, 32671,  4827,   443,  2300, 14609,
                                         18295,  5035,   346, 18237,     3,  1996,   327,    15,   187,   187,
                                         36, 32671,  4827, 35645,  8100, 18295,  5035,   346, 18237,     3,
                                         1996,   327,    15,   187,   187,  2573, 30741, 18295,    34,  5112,
                                         3159,    13,  4495,   346, 33729,   937,  4536,   908,   407,  4383,
                                         187, 34782,   327, 29992,    15,   187,   187,  2573,  6766, 18295,
                                         34,  5112,  3159,  4495,   346, 48781,   937,   285,  4536,   908],
                                       [  187,  2773,   253, 28245,   373,   285, 29555,   262, 22348,   373,
                                          497,   767,  1027, 16308,   273,   187,   264,   692,   265,    13,
                                          369,  1620, 17801,    13,   253,  3438,  1146,  4270,   275,   253,
                                          5281,   273,   247,   187,  6017,   280,  1426,   282,    28,   253,
                                          643,  3839, 32679,    13,   594,   347,   281,  1056,   253,  1072,
                                          4677,   347,   187,   338,   767, 28245,   373,   943,   320,  7416,
                                          2366,    15,  9110,   253,  1072,  1659,   310,  2223,   187,  8890,
                                          407,   841,  4454,   275,  2067,  4477,    15,  1583,  1646,    13,
                                          1512,    13,   281,   452,   644,   187, 38061,   323,  3240,  1027,
                                          7637,    27,   253, 28245,   373,   323,  3924,  7120,    13,   253],
                                       [   13,  1014,  2167,   344,  6057,   326,   253, 43412,   556,   816,
                                           35517,   521,   187, 13875,    27,   187,   187,     3,   688,   253,
                                           1083,  1411,   368,    13,  2305,    15,  3619, 20111,    13,   275,
                                           634,  5928,   309,   452,  6507,   314,   187,  3354,   728,   253,
                                           4179,    15,   496,   253,  1083,  1411, 18682,   659,  7352,  1634,
                                           386,   309, 13292,   512,   187, 36871,   273,    13,   285, 10343,
                                           281,  3890,   271,  4743,  1919,   309,   452,   574,   271,   187,
                                           10468, 32268,   273,  2819,   715,    13,   253,  5989,   273,   253,
                                           5575, 34306,   449,   187,   187,  3996, 20111,   665,   574,  6225,
                                           1309,   436,  2427,   689,   285,  2335,   247,  7319,   387,   253]], dtype=torch.int64, device="cpu"),
            "index": torch.tensor([24604650, 44278198, 10199766, 29398698, 30323890, 26381103,  6706633, 44120804], dtype=torch.int64, device="cpu")
        }

    optim.zero_grad(set_to_none=True)
    batch = move_to_device(batch, device)
    micro_batch = {
        key: value[:2] for key, value in batch
    }

    ce_batch_loss = torch.tensor(0.0, device=device)
    with torch.autocast("cuda", enabled=True, dtype=cfg.autocast_precision):
        # Run forward pass.
        logits = fsdp_model(
            input_ids=micro_batch["input_ids"],
            attention_mask=micro_batch.get("attention_mask"),
            attention_bias=micro_batch.get("attention_bias"),
        ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = micro_batch["input_ids"]
        labels = labels[..., 1:].contiguous()
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(logits_for_loss, labels, ignore_index=-100, reduction="mean")
        ce_loss = ce_loss / len(micro_batches)

        # In case this helps with memory utilization.
        del micro_batch

        # Update overall CE batch loss.
        ce_batch_loss += ce_loss.detach()

        # Get loss to optimize for.
        loss = ce_loss

        del logits

    # Check for nan.
    if torch.isnan(loss):
        raise ValueError("nan loss encountered")

    # Run backward pass.
    log.info("Before new backward")
    loss.backward()
    log.info("After new backward")


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
