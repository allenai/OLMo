import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import olmo
from olmo import Olmo, TrainConfig
from olmo.alt_rope_models import OlmoComplexRope, OlmoTorchScriptRope
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


def forward_benchmark(model, batch):
    # Warmup
    for _ in range(10):
        model(batch)
    # Benchmark
    benchmark_results = benchmark.Timer(
        stmt="model(batch)",
        globals={"model": model, "batch": batch},
        num_threads=1,
        num_iters=10,
        label="olmo",
    ).blocked_autorange(min_run_time=1)
    return benchmark_results


def forward_backward_benchmark(model, batch):
    def forward_backward():
        output = model(batch)
        loss = F.cross_entropy(output, batch[:, 0])
        loss.backward()

    # Warmup
    for _ in range(10):
        forward_backward()
    # Benchmark
    benchmark_results = benchmark.Timer(
        stmt="forward_backward()",
        globals={"model": model, "batch": batch, "forward_backward": forward_backward},
        num_threads=1,
        num_iters=10,
        label="olmo",
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

    with open("scripts/spiky_batch.npy", "rb") as f:
        buffer = f.read()
    array = np.frombuffer(buffer, dtype=np.uint64)
    batch = torch.tensor(array.astype(np.int_), dtype=torch.long)
    batch = batch.reshape(2048, -1).to(device)

    log.info("Running forward benchmark on base model...")
    log.info(forward_benchmark(olmo_base, batch))
    log.info("Running forward benchmark on TorchScript model...")
    log.info(forward_benchmark(olmo_torchscript, batch))
    log.info("Running forward benchmark on complex model...")
    log.info(forward_benchmark(olmo_complex, batch))

    log.info("Running forward-backward benchmark on base model...")
    log.info(forward_backward_benchmark(olmo_base, batch))
    log.info("Running forward-backward benchmark on TorchScript model...")
    log.info(forward_backward_benchmark(olmo_torchscript, batch))
    log.info("Running forward-backward benchmark on complex model...")
    log.info(forward_backward_benchmark(olmo_complex, batch))


if __name__ == "__main__":
    main()
