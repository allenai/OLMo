import os

import torch
import torch.distributed as dist


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def print_rank0(*args):
    if dist.get_rank() == 0:
        print("RANK 0", *args)


def main() -> None:
    dist.init_process_group(backend="nccl")
    dist.barrier()
    print_rank0("Distributed process group initialized")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    dist.barrier()
    print_rank0("Done!")


if __name__ == "__main__":
    main()
