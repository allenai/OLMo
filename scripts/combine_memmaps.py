"""
Use this script to combine many numpy memory-mapped files from a train config into one.
"""

import logging
import sys

import numpy as np

from olmo.config import DataConfig
from olmo.exceptions import OlmoCliError
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def main(cfg: DataConfig, output_path: str) -> None:
    paths = cfg.paths
    assert paths
    mmaps = [np.memmap(path, mode="r", dtype=np.uint16) for path in paths]
    total_size = sum(mmap.shape[0] for mmap in mmaps)

    log.info(f"Combining {len(paths)} memory-mapped data files into {output_path} with {total_size:,d} items...")
    out = np.memmap(output_path, mode="w+", dtype=np.uint16, shape=(total_size,))
    offset = 0
    for mmap in mmaps:
        out[offset : offset + mmap.shape[0]] = mmap
        offset += mmap.shape[0]
    out.flush()
    log.info("Done!")


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, output_path = sys.argv[1], sys.argv[2]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OUTPUT_PATH]")

    cfg = DataConfig.load(yaml_path, key="data")
    main(cfg, output_path)
