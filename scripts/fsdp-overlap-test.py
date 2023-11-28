import logging
from pathlib import Path
from typing import List, Optional
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
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)

RANK_TO_BATCH_SIZE_MAP = {
    0: 2 ** 3,
    1: 0,
}
PARAM_DIM: int = 2 ** 13
GATHER_DIM: int = 2 ** 14


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.param1 = torch.nn.Linear(PARAM_DIM, PARAM_DIM)
        # self.param2 = torch.nn.Linear(PARAM_DIM, PARAM_DIM)
        # self.param3 = torch.nn.Linear(PARAM_DIM, PARAM_DIM)
        # self.param4 = torch.nn.Linear(PARAM_DIM, PARAM_DIM)
        # self.param5 = torch.nn.Linear(PARAM_DIM, PARAM_DIM)

        self.params = torch.nn.ModuleList([
            torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
            torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
            torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
            torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
            torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
        ])

    def forward(self, x):
        for param in self.params:
            x = param(x)

        return x


def get_profiler(save_folder: str) -> torch.profiler.profile:
    profiling_schedule = schedule(wait=0, warmup=5, active=5)

    def on_trace_ready(p):
        profiler_output_dir = Path(save_folder) / "profiler"
        profiler_output_dir.mkdir(exist_ok=True)

        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
        log.info("Profile by total GPU time at step %d:\n%s", p.step_num, output)
        output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
        log.info("Profile by total CPU time at step %d:\n%s", p.step_num, output)

        p.export_chrome_trace(str((profiler_output_dir / f"{get_global_rank()}.{p.step_num}.chrome_trace.json.gz")))

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
        sync_module_states=True,  # Ensure same init across all ranks for this test
        sharding_strategy=ShardingStrategy.FULL_SHARD,
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


def do_communication(data_to_gather: torch.Tensor, gather_list: Optional[List[torch.Tensor]]):
    dist.gather(data_to_gather, gather_list, 1)


def do_computation(fsdp_model: FSDP, batch: torch.Tensor):
    fsdp_model(batch)


def run_batch(fsdp_model: FSDP, batch: torch.Tensor, data_to_gather: torch.Tensor, gather_list: Optional[List[torch.Tensor]]):
    do_computation(fsdp_model, batch)
    do_communication(data_to_gather, gather_list)
    barrier()


def run_batches(fsdp_model: FSDP):
    batch_size = RANK_TO_BATCH_SIZE_MAP[get_global_rank()]

    save_folder = "."
    torch_profiler = get_profiler(save_folder)

    batch = torch.randn((batch_size, PARAM_DIM)).cuda()
    data_to_gather = torch.randn((GATHER_DIM, GATHER_DIM)).cuda()
    gather_list = None
    if get_global_rank() == 1:
        gather_list = [torch.zeros((GATHER_DIM, GATHER_DIM)).cuda(), torch.zeros((GATHER_DIM, GATHER_DIM)).cuda()]

    with torch_profiler as p:
        for _ in range(10):
            run_batch(fsdp_model, batch, data_to_gather, gather_list)
            p.step()


def test():
    fsdp_model = get_fsdp_model()

    log.info("Peak GPU Memory (MB) after FSDP: %d", int(peak_gpu_memory() or 0))
    log.info("Model:")
    log.info(fsdp_model)

    # log.warning("Param 1 weight shape %s", fsdp_model.param1.weight.shape)
    # log.warning("Param 2 weight shape %s", fsdp_model.param2.weight.shape)
    # log.warning("Param weight shape %s", fsdp_model.param.weight.shape)
    for param in fsdp_model.parameters():
        log.warning("Param weight shape %s", param.shape)
    log.info("Global rank %d", get_global_rank())
    log.info("Local rank %d", get_local_rank())

    run_batches(fsdp_model)


def main():
    test()


if __name__ == "__main__":
    init_process_group()
    prepare_cli_environment()
    main()
