import argparse
import logging
import os
import re

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import ShardingStrategy

from olmo import (
    ActivationType,
    DDPConfig,
    InitFnType,
    LayerNormType,
    ModelConfig,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TokenizerConfig,
    TrainConfig,
    WandbConfig,
)
from olmo.config import (
    DataConfig,
    DistributedStrategy,
    EvaluatorConfig,
    EvaluatorType,
    FSDPConfig,
    FSDPPrecision,
    FSDPWrapStrategy,
    InstanceFilterConfig,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
)
from olmo.data import named_data_mixes
from olmo.util import add_cached_path_clients, prepare_cli_environment

log = logging.getLogger("train")

MODEL_CONFIG_150M = ModelConfig(
    d_model=768,
    n_heads=12,
    n_layers=12,
    mlp_ratio=8,
    weight_tying=False,
    alibi=False,
    rope=True,
    flash_attention=True,
    attention_dropout=0.0,
    attention_layer_norm=False,
    include_bias=False,
    layer_norm_type=LayerNormType.rms,
    layer_norm_with_affine=True,
    layer_norm_eps=1e-6,
    bias_for_layer_norm=False,
    attention_layer_norm_with_affine=False,
    activation_type=ActivationType.swiglu,
    residual_dropout=0.0,
    embedding_dropout=0.0,
    max_sequence_length=2048,
    vocab_size=50280,
    embedding_size=50304,
    eos_token_id=0,
    pad_token_id=1,
    init_device="cuda",
    init_fn=InitFnType.normal,
    init_std=0.02,
    init_cutoff_factor=3,
)

MODEL_CONFIGS = {
    "150M": MODEL_CONFIG_150M,
    "300M": MODEL_CONFIG_150M.update_with(d_model=1024, n_heads=16, n_layers=16, mlp_ratio=8),
    "750M": MODEL_CONFIG_150M.update_with(d_model=1536, n_heads=16, n_layers=16, mlp_ratio=8),
    "1B": MODEL_CONFIG_150M.update_with(d_model=2048, n_heads=16, n_layers=16, mlp_ratio=8),
    "7B": MODEL_CONFIG_150M.update_with(
        d_model=4096, n_heads=32, n_layers=32, mlp_ratio=None, mlp_hidden_size=22016, init_device="meta"
    ),
}


_number_unit_re = re.compile(r"^([0-9]+)([a-zA-Z]+)$")


def size_cmd(args: argparse.Namespace):
    raise NotImplementedError()


def train_cmd(args: argparse.Namespace):
    # Construct a config
    args.model = args.model.strip().upper()
    run_name = f"{args.name}-{args.model}-{args.length}"
    assert "/" not in run_name

    permanent_data_prefix = "/weka/oe-training-default/ai2-llm"
    if args.s3:
        permanent_data_prefix = "s3://ai2-llm"
    permanent_data_prefix.rstrip("/")
    if args.write_location is None:
        if permanent_data_prefix.startswith("s3://"):
            save_folder = f"runs/{run_name}"
            remote_save_folder = f"{permanent_data_prefix}/checkpoints/OLMo-ladder/{run_name}"
        else:
            save_folder = f"{permanent_data_prefix}/checkpoints/OLMo-ladder/{run_name}"
            remote_save_folder = None
    else:
        save_folder = args.write_location
        remote_save_folder = None

    model_config = MODEL_CONFIGS[args.model]

    model_size, model_size_unit = _number_unit_re.match(args.model).groups()  # type: ignore
    model_size = int(model_size)
    if model_size_unit == "K":
        model_size *= 1000
    elif model_size_unit == "M":
        model_size *= 1000000
    elif model_size_unit == "B":
        model_size *= 1000000000
    else:
        raise ValueError(f"Could not parse model name '{args.model}'")
    del model_size_unit

    length_in_tokens, length_unit = _number_unit_re.match(args.length.strip().upper()).groups()  # type: ignore
    length_in_tokens = int(length_in_tokens)
    if length_unit == "C" or length_unit == "XC":
        length_in_tokens *= 20 * model_size
    elif length_unit == "K":
        length_in_tokens *= 1000
    elif length_unit == "M":
        length_in_tokens *= 1000000
    elif length_unit == "B":
        length_in_tokens *= 1000000000
    elif length_unit == "T":
        length_in_tokens *= 1000000000000
    else:
        raise ValueError(f"Could not parse length '{args.length}'")
    del length_unit

    # calculate batch size according to
    # https://www.semanticscholar.org/reader/5585191b1b479346ecf173be3b35c8313b77d457
    # holds only for a sequence length of 2048 (but could probably be easily adapted)
    assert model_config.max_sequence_length == 2048
    global_batch_size = 160 * (model_size / 108000000) ** (2 / 3)
    global_batch_size /= 8 * 4  # 8 GPUs per node, microbatch size 4
    global_batch_size = round(global_batch_size)
    global_batch_size *= 8 * 4

    # We don't want the global batch size depend on the device batch size, because we might have to change the
    # device batch size based on the hardware we're running on.
    device_batch_size = {
        "150M": 16,
        "1B": 2,
        "7B": 2,
    }.get(args.model, 4)

    assert global_batch_size % device_batch_size == 0

    # calculate the learning rate according to the same paper
    lr = 0.0047 * (model_size / 108000000) ** (-1 / 3)

    save_interval = {
        "1B": 2500,
        "7B": 1000,
    }.get(args.model, 5000)

    distributed_strategy = {"7B": DistributedStrategy.fsdp}.get(args.model, DistributedStrategy.ddp)

    cfg = TrainConfig(
        run_name=run_name,
        seed=6198,
        wandb=None if not args.wandb else WandbConfig(name=run_name, project="olmo-ladder"),
        model=model_config,
        ddp=DDPConfig(),  # defaults are fine
        fsdp=FSDPConfig(
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.mixed,
        ),
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            learning_rate=lr,
            weight_decay=0.05,
            eps=1e-8,
            decay_norm_and_bias=True,
            decay_embeddings=True,
            betas=(0.9, 0.95),
            metrics_log_interval=10,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.cosine_with_warmup,
            alpha_f=0.01,
            warmup_min_lr=0.0,
            t_warmup=round(model_size / (global_batch_size * model_config.max_sequence_length)),
        ),
        max_duration=f"{length_in_tokens}T",
        global_train_batch_size=global_batch_size,
        tokenizer=TokenizerConfig(identifier="tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"),
        save_folder=save_folder,
        remote_save_folder=remote_save_folder,
        save_overwrite=args.save_overwrite,
        save_interval_unsharded=save_interval,
        save_num_unsharded_checkpoints_to_keep=-1,
        save_interval=None,
        sharded_checkpointer=ShardedCheckpointerType.olmo_core,
        device_train_microbatch_size=device_batch_size,
        precision="amp_bf16",
        distributed_strategy=distributed_strategy,
        fused_loss=True,
        max_grad_norm=1.0,
        speed_monitor=SpeedMonitorConfig(window_size=1),
        eval_interval=save_interval,
        device_eval_batch_size=device_batch_size,
        evaluators=[
            EvaluatorConfig(
                label="all-small-ppl-validation",
                data=DataConfig(
                    drop_last=True,
                    datasets={
                        "c4_en-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/c4_en/val/part-0-00000.npy"
                        ],
                        "dolma_books-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/dolma_books/val/part-0-00000.npy"
                        ],
                        "dolma_common-crawl-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/dolma_common-crawl/val/part-0-00000.npy"
                        ],
                        "dolma_pes2o-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/dolma_pes2o/val/part-0-00000.npy"
                        ],
                        "dolma_reddit-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/dolma_reddit/val/part-0-00000.npy"
                        ],
                        "dolma_stack-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/dolma_stack/val/part-0-00000.npy"
                        ],
                        "dolma_wiki-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/dolma_wiki/val/part-0-00000.npy"
                        ],
                        "ice-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/ice/val/part-0-00000.npy"
                        ],
                        "m2d2_s2orc-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/m2d2_s2orc/val/part-0-00000.npy"
                        ],
                        "pile-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/pile/val/part-0-00000.npy"
                        ],
                        "wikitext_103-validation": [
                            f"{permanent_data_prefix}/eval-data/perplexity/v3_small_gptneox20b/wikitext_103/val/part-0-00000.npy"
                        ],
                    },
                ),
            ),
            EvaluatorConfig(label="piqa", type=EvaluatorType.downstream),
            EvaluatorConfig(label="hellaswag", type=EvaluatorType.downstream),
            EvaluatorConfig(label="winogrande", type=EvaluatorType.downstream),
            EvaluatorConfig(label="openbook_qa", type=EvaluatorType.downstream),
            EvaluatorConfig(label="boolq", type=EvaluatorType.downstream),
            EvaluatorConfig(label="sciq", type=EvaluatorType.downstream),
            EvaluatorConfig(label="arc_easy", type=EvaluatorType.downstream),
            EvaluatorConfig(label="arc_challenge", type=EvaluatorType.downstream),
            EvaluatorConfig(label="copa", type=EvaluatorType.downstream),
            EvaluatorConfig(label="commonsense_qa", type=EvaluatorType.downstream),
            EvaluatorConfig(label="social_iqa", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_stem_var", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_humanities_var", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_social_sciences_var", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_other_var", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_stem_mc_5shot", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_humanities_mc_5shot", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_social_sciences_mc_5shot", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_other_mc_5shot", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_stem_mc_5shot_test", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_humanities_mc_5shot_test", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_social_sciences_mc_5shot_test", type=EvaluatorType.downstream),
            EvaluatorConfig(label="mmlu_other_mc_5shot_test", type=EvaluatorType.downstream),
            EvaluatorConfig(label="basic_arithmetic", type=EvaluatorType.downstream),
            EvaluatorConfig(label="trivia_qa_wiki_ppl", type=EvaluatorType.downstream),
            EvaluatorConfig(label="natural_qs_open_ppl", type=EvaluatorType.downstream),
            EvaluatorConfig(label="arc_easy_ppl", type=EvaluatorType.downstream),
        ],
        data=DataConfig(
            num_workers=32,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
            instance_filter=InstanceFilterConfig(),  # defaults are fine
            paths=[f"{permanent_data_prefix}/{path}" for path in named_data_mixes.DATA_PATHS[args.data]],
        ),
    )

    # Do a bunch of initialization
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    dist.init_process_group(backend="nccl")
    prepare_cli_environment()
    add_cached_path_clients()

    from train import main

    main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(os.path.basename(__file__))
    subparsers = parser.add_subparsers(required=True)

    size_parser = subparsers.add_parser("size")
    size_parser.set_defaults(func=size_cmd)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", type=str, required=True)
    train_parser.add_argument("--data", type=str, required=True)
    train_parser.add_argument("--length", type=str, default="2xC")
    train_parser.add_argument("--name", type=str, required=True)
    train_parser.add_argument(
        "--s3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="read data from S3, write checkpoints to S3",
    )
    train_parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True, help="create a run in wandb"
    )
    train_parser.add_argument("--write_location", type=str, default=None)
    train_parser.add_argument("--save_overwrite", action="store_true")
    train_parser.set_defaults(func=train_cmd)

    args = parser.parse_args()
    args.func(args)
