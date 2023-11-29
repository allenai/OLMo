import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
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
    1: 2 ** 3,
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


def init_process_group():
    # Initialize process group and set device.
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")


def do_communication(data_to_gather: torch.Tensor, gather_list: Optional[List[torch.Tensor]]):
    s: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(s):
        dist.gather(data_to_gather, gather_list, 1)


def do_computation(model: Model, batch: torch.Tensor):
    s: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(s):
        model(batch)


def run_batch(model: Model, batch: torch.Tensor, data_to_gather: torch.Tensor, gather_list: Optional[List[torch.Tensor]]):
    do_computation(model, batch)
    do_communication(data_to_gather, gather_list)
    barrier()


def run_batches(model: Model):
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
            run_batch(model, batch, data_to_gather, gather_list)
            p.step()


def test():
    model = Model()

    log.info("Peak GPU Memory (MB) after FSDP: %d", int(peak_gpu_memory() or 0))
    log.info("Model:")
    log.info(model)

    # log.warning("Param 1 weight shape %s", model.param1.weight.shape)
    # log.warning("Param 2 weight shape %s", model.param2.weight.shape)
    # log.warning("Param weight shape %s", model.param.weight.shape)
    for param in model.parameters():
        log.warning("Param weight shape %s", param.shape)
    log.info("Global rank %d", get_global_rank())
    log.info("Local rank %d", get_local_rank())

    run_batches(model)


def main():
    test()


if __name__ == "__main__":
    init_process_group()
    prepare_cli_environment()
    main()
