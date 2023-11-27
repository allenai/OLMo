import logging
from pathlib import Path
from packaging import version

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.profiler import ProfilerActivity, schedule

from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)

log = logging.getLogger(__name__)

RANK_TO_BATCH_SIZE_MAP = {
    0: 32,
    1: 16384,
}
INPUT_DIM: int = 1024


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.param = torch.nn.Linear(INPUT_DIM, 32768)

    def forward(self, x):
        self.param(x)


def get_profiler(save_folder: str) -> torch.profiler.profile:
    profiling_schedule = schedule(wait=0, warmup=0, active=1)

    def on_trace_ready(p):
        profiler_output_dir = Path(save_folder) / "profiler"
        profiler_output_dir.mkdir(exist_ok=True)

        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
        log.info("Profile by total GPU time at step %d:\n%s", p.step_num, output)
        output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
        log.info("Profile by total CPU time at step %d:\n%s", p.step_num, output)

        p.export_chrome_trace(str((profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz")))

    return torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        schedule=profiling_schedule,
        on_trace_ready=on_trace_ready,
    )


def get_fsdp_model() -> FSDP:
    # Initialize the model.
    log.info("Building model...")
    model = Model()
    log.info("Peak GPU Memory (MB) before FSDP: %d", int(peak_gpu_memory() or 0))

    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=get_default_device())

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=None,
        use_orig_params=True,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        param_init_fn=param_init_fn,
    )


def init_process_group():
    # Initialize process group and set device.
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")


def test():
    init_process_group()
    fsdp_model = get_fsdp_model()

    log.info("Peak GPU Memory (MB) after FSDP: %d", int(peak_gpu_memory() or 0))
    log.info("Model:")
    log.info(fsdp_model)

    batch_size = RANK_TO_BATCH_SIZE_MAP[get_global_rank()]

    for _ in range(10):
        x = torch.randn((batch_size, INPUT_DIM))
        fsdp_model(x)


def main():
    save_folder = "."
    torch_profiler = get_profiler(save_folder)
    with torch_profiler as p:
        test()


if __name__ == "__main__":
    main()
