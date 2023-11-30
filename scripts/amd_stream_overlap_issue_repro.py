import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.cuda import Stream
from torch.profiler import ProfilerActivity, schedule

RANK_TO_BATCH_SIZE_MAP = {
    0: 2**11,
    1: 2**11,
}
PARAM_DIM: int = 2**13
GATHER_DIM: int = 2**14


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.params = torch.nn.ModuleList(
            [
                torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
                torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
                torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
                torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
                torch.nn.Linear(PARAM_DIM, PARAM_DIM, bias=False),
            ]
        )

    def forward(self, x):
        for param in self.params:
            x = param(x)

        return x


def get_global_rank() -> int:
    return int(os.environ.get("RANK") or dist.get_rank())


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_profiler(save_folder: str) -> torch.profiler.profile:
    profiling_schedule = schedule(wait=0, warmup=5, active=1)

    def on_trace_ready(p):
        profiler_output_dir = Path(save_folder) / "profiler"
        profiler_output_dir.mkdir(exist_ok=True)

        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
        print(f"Profile by total GPU time at step {p.step_num}:\n{output}")
        output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
        print(f"Profile by total CPU time at step {p.step_num}:\n{output}")

        p.export_chrome_trace(
            str((profiler_output_dir / f"{get_global_rank()}.{p.step_num}.chrome_trace.json.gz"))
        )

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


def do_communication(data_to_gather: torch.Tensor, gather_list: Optional[List[torch.Tensor]], stream: Optional[Stream] = None):
    # if stream is None:
    #     stream = Stream()

    with torch.cuda.stream(stream):
        # dist.gather(data_to_gather, gather_list, 1)
        dist.all_gather(gather_list, data_to_gather)


def do_computation(model: Model, batch: torch.Tensor, stream: Optional[Stream] = None):
    # if stream is None:
    #     stream = Stream()

    with torch.cuda.stream(stream):
        model(batch)


def run_batch(
    model: Model,
    batch: torch.Tensor,
    data_to_gather: torch.Tensor,
    gather_list: Optional[List[torch.Tensor]],
    communication_stream: Optional[Stream] = None,
    computation_stream: Optional[Stream] = None,
):
    do_computation(model, batch, computation_stream)
    do_communication(data_to_gather, gather_list, communication_stream)
    barrier()


def run_batches(model: Model):
    batch_size = RANK_TO_BATCH_SIZE_MAP[get_global_rank()]

    save_folder = "."
    torch_profiler = get_profiler(save_folder)

    batch = torch.randn((batch_size, PARAM_DIM)).cuda()
    data_to_gather = torch.randn((GATHER_DIM, GATHER_DIM)).cuda()

    gather_list = [torch.zeros((GATHER_DIM, GATHER_DIM)).cuda(), torch.zeros((GATHER_DIM, GATHER_DIM)).cuda()]

    communication_stream: Optional[Stream] = Stream()
    computation_stream: Optional[Stream] = None

    with torch_profiler as p:
        for _ in range(6):
            run_batch(model, batch, data_to_gather, gather_list, communication_stream, computation_stream)

            # Print an element from every tensor to force device synchronization
            # (just in case).
            print("Tensor first elements:", batch[0, 0], data_to_gather[0, 0], gather_list[0][0, 0], gather_list[1][0, 0])

            p.step()


def test():
    model = Model().cuda()

    print("Model:")
    print(model)

    for param in model.parameters():
        print(f"Param weight shape {param.shape}")
    print(f"Global rank {get_global_rank()}")
    print(f"Local rank {get_local_rank()}")

    run_batches(model)


def main():
    test()


if __name__ == "__main__":
    init_process_group()
    main()
