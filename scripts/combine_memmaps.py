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
    log.info(f"Reading memmaps to find total size...")
    total_size = 0
    for path in paths:
        mmap = np.memmap(path, mode="r", dtype=np.uint16)
        total_size += mmap.shape[0]

    log.info(f"Combining {len(paths)} memory-mapped data files into {output_path} with {total_size:,d} items...")
    out = np.memmap(output_path, mode="w+", dtype=np.uint16, shape=(total_size,))
    offset = 0
    for path in paths:
        mmap = np.memmap(path, mode="r", dtype=np.uint16)
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
