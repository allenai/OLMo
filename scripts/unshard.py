import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Union

import torch
from torch import Tensor
from torch.distributed import FileStore, init_process_group
from torch.distributed._shard.sharded_tensor import ShardedTensor

logger = logging.getLogger(__name__)


def _unshard_worker(shard_count: int, shard: Path, output_dir: Path):
    shard_number = int(shard.name[4:-3])  # shard names look like "rankXX.pt"
    init_process_group(world_size=shard_count, rank=shard_number, init_method='tcp://127.0.0.1:32323', backend="gloo")

    logger.info("Loading %s ...", shard.name)
    state_dict = torch.load(shard, map_location="cpu")
    logger.info("Loaded %s", shard.name)

    def unshard_tensor(t: ShardedTensor) -> Tensor:
        if shard_number == 0:
            out = torch.zeros(*t.shape, dtype=t.dtype)
        else:
            out = None
        t.gather(out=out)
        return out

    def unshard_object(o: Any) -> Any:
        if isinstance(o, str):
            return o
        elif isinstance(o, (list, tuple, set)):
            return o.__class__(unshard_object(i) for i in o)
        elif isinstance(o, dict):
            return {key: unshard_object(value) for key, value in o.items()}
        elif isinstance(o, ShardedTensor):
            return unshard_tensor(o)
        else:
            return o

    # model
    if shard_number == 0:
        logger.info("Unsharding model state ...")
    model_state_dict = unshard_object(state_dict.pop("model"))
    if shard_number == 0:
        logger.info("Unsharded model state")
        model_output = str(output_dir / "model.pt")
        logger.info("Saving model state to %s", model_output)
        torch.save(model_state_dict, model_output)
    del model_state_dict

    # optimizer
    if shard_number == 0:
        logger.info("Unsharding optimizer state ...")
    optim_state_dict = unshard_object(state_dict.pop("optim"))
    if shard_number == 0:
        logger.info("Unsharded optimizer state")
        optim_output = str(output_dir / "optim.pt")
        logger.info("Saving optimizer state to %s", optim_output)
        torch.save(optim_state_dict, optim_output)
    del optim_state_dict

    # whatever is left
    if shard_number == 0:
        logger.info("Unsharding everything else ...")
    other_state_dict = unshard_object(state_dict)
    if shard_number == 0:
        logger.info("Unsharded everything else")
        other_output = str(output_dir / "other.pt")
        logger.info("Saving everything else to %s", other_output)
        torch.save(other_state_dict, other_output)
    del other_state_dict


def unshard(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This environment variable needs to be set on LUMI in the workers.
    os.environ["OMP_NUM_THREADS"] = "1"

    shards = list(input_dir.glob("rank*.pt"))
    if len(shards) <= 0:
        raise RuntimeError(f"Could not find any shards at {input_dir}")
    pids = []
    for shard in shards:
        pid = os.fork()
        if pid == 0:
            _unshard_worker(len(shards), shard, output_dir)
            sys.exit(0)
        else:
            pids.append(pid)
    logger.info("Launched %s workers with these pids: %s", len(pids), " ".join(str(pid) for pid in pids))

    logger.info("Copying config.yaml to %s", output_dir)
    shutil.copy(input_dir / "config.yaml", output_dir)

    logger.info("Waiting for workers to finish ...")
    for pid in pids:
        _, retval = os.waitpid(pid, 0)
        if retval != 0:
            raise RuntimeError("Child process returned non-zero status code.")
    logger.info("Workers finished")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: unshard.py <input dir> <output dir>")
        sys.exit(1)
    else:
        logging.basicConfig(level=logging.INFO)
        unshard(sys.argv[1], sys.argv[2])
