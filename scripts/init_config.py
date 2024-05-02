"""
Run this to initialize a new training config to a file.
"""
import logging
import sys
from pathlib import Path
from typing import List

from olmo import TrainConfig
from olmo.exceptions import OLMoCliError
from olmo.util import clean_opt, prepare_cli_environment

log = logging.getLogger(__name__)


def main(save_path: Path, args_list: List[str]) -> None:
    cfg = TrainConfig.new(overrides=args_list)
    log.info("Configuration:")
    log.info(cfg)
    cfg.save(save_path)
    log.info(f"Config saved to {save_path}")


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        save_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [SAVE_PATH] [OPTIONS]")

    main(Path(save_path), [clean_opt(s) for s in args_list])
