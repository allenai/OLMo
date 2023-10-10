import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import olmo
from olmo import Olmo, TrainConfig, RotaryEmbedding
from olmo.alt_rope_models import (
    OlmoComplexRope,
    OlmoTorchScriptRope,
    RotaryEmbeddingTorchScripted,
    ComplexRotaryEmbedding,
)
from olmo.util import prepare_cli_environment

prepare_cli_environment()
log = logging.getLogger(__name__)


def get_world_size():
    return 1


def build_olmo_model(complex: bool, torch_script: bool, model_path: str, device: str):
    cfg = TrainConfig.load(model_path)
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    cfg.model.init_device = device
    cfg.model.n_layers = 2

    # Make model
    log.info("Building model...")
    if not complex:
        if not torch_script:
            olmo_model = Olmo(cfg.model)
        else:
            olmo_model = OlmoTorchScriptRope(cfg.model)
    else:
        if not torch_script:
            olmo_model = OlmoComplexRope(cfg.model)
        else:
            raise NotImplementedError("Complex model with TorchScript is not implemented yet")
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    return olmo_model.to(device)


def rope_forward_benchmark(rope_embed, q, k, label):
    # Warmup
    for _ in range(10):
        rope_embed(q, k)
    # Benchmark
    benchmark_results = benchmark.Timer(
        stmt="rope_embed(q, k)",
        globals={"rope_embed": rope_embed, "q": q, "k": k},
        num_threads=1,
        label=label,
    ).blocked_autorange(min_run_time=1)
    return benchmark_results


def rope_forward_backward_benchmark(rope_embed, q, k, label):
    def forward_backward():
        q_out, k_out = rope_embed(q, k)
        loss = q_out.sum() + k_out.sum()
        loss.backward(retain_graph=True)

    # Warmup
    for _ in range(10):
        forward_backward()
    # Benchmark
    benchmark_results = benchmark.Timer(
        stmt="forward_backward()",
        globals={"rope_embed": rope_embed, "q": q, "k": k, "forward_backward": forward_backward},
        num_threads=1,
        label=label,
    ).blocked_autorange(min_run_time=1)
    return benchmark_results


def model_forward_benchmark(model, batch, label):
    # Warmup
    for _ in range(10):
        model(batch)
    # Benchmark
    benchmark_results = benchmark.Timer(
        stmt="model(batch)",
        globals={"model": model, "batch": batch},
        num_threads=1,
        label=label,
    ).blocked_autorange(min_run_time=1)
    return benchmark_results


def model_forward_backward_benchmark(model: Olmo, batch: torch.Tensor, run_label):
    def convert_label_to_look_like_logits(batch):
        one_hot_labels = torch.zeros(
            batch.shape[0], model.config.embedding_size, device=batch.device, dtype=torch.long
        )
        for i, label in enumerate(batch):
            one_hot_labels[i, label] = 1
        return one_hot_labels

    def forward_backward():
        output = model(batch).logits
        loss = F.cross_entropy(output, convert_label_to_look_like_logits(batch[:, 0]))
        loss.backward()

    # Warmup
    for _ in range(10):
        forward_backward()
    # Benchmark
    benchmark_results = benchmark.Timer(
        stmt="forward_backward()",
        globals={"model": model, "batch": batch, "forward_backward": forward_backward},
        num_threads=1,
        label=run_label,
    ).blocked_autorange(min_run_time=1)
    return benchmark_results


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # needed for running in the deterministic mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    model_path = "configs/v1_5-mix-medium-mitch-ish.yaml"

    olmo_base = build_olmo_model(complex=False, torch_script=False, model_path=model_path, device=device)
    olmo_torchscript = build_olmo_model(complex=False, torch_script=True, model_path=model_path, device=device)
    olmo_complex = build_olmo_model(complex=True, torch_script=False, model_path=model_path, device=device)

    rope_base = olmo_base.transformer.blocks[0].rotary_emb
    rope_torchscript = olmo_torchscript.transformer.blocks[0].rotary_emb
    rope_complex = olmo_complex.transformer.blocks[0].rotary_emb.to(device)

    with open("scripts/spiky_batch.npy", "rb") as f:
        buffer = f.read()
    array = np.frombuffer(buffer, dtype=np.uint64)
    batch = torch.tensor(array.astype(np.int_), dtype=torch.long)
    batch = batch.reshape(2048, -1)[:50]  # don't run all 4M tokens
    batch = batch.to(device)
    q, k = olmo_base.forward_right_up_to_rope(batch)
    q, k = q.to(device), k.to(device)

    log.info("Running forward benchmark on base model...")
    log.info(rope_forward_benchmark(rope_base, q, k, "base-rope-forward"))
    log.info("Running forward benchmark on TorchScript model...")
    log.info(rope_forward_benchmark(rope_torchscript, q, k, "torchscript-rope-forward"))
    log.info("Running forward benchmark on complex model...")
    log.info(rope_forward_benchmark(rope_complex, q.transpose(1,2), k.transpose(1,2), "complex-rope-forward"))

    log.info("Running forward-backward benchmark on base model...")
    log.info(rope_forward_backward_benchmark(rope_base, q, k, "base-rope-forward-backward"))
    log.info("Running forward-backward benchmark on TorchScript model...")
    log.info(rope_forward_backward_benchmark(rope_torchscript, q, k, "torchscript-rope-forward-backward"))
    log.info("Running forward-backward benchmark on complex model...")
    log.info(rope_forward_backward_benchmark(rope_complex, q.transpose(1,2), k.transpose(1,2), "complex-rope-forward-backward"))


if __name__ == "__main__":
    main()
