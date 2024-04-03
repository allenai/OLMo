"""
Run this script to quickly determine how big a model is for a given training configuration file.

For example:

```bash
python scripts/show_model_size.py train_config.yaml
```
"""
import logging
import sys

from olmo import OLMo, TrainConfig
from olmo.exceptions import OLMoCliError
from olmo.util import clean_opt, prepare_cli_environment

log = logging.getLogger(__name__)


def main(cfg: TrainConfig) -> None:
    cfg.model.init_device = "cpu"

    n_layers = cfg.model.n_layers
    cfg.model.n_layers = 1

    single_layer_model = OLMo(cfg.model)
    block = single_layer_model.transformer.blocks[0]  # type: ignore
    params_per_block = sum(p.numel() for p in block.parameters())  # type: ignore

    log.info(
        f"Total number of parameters: {single_layer_model.num_params() + (params_per_block * (n_layers - 1)):,d}"
    )
    log.info(
        f"Number of non-embedding parameters: "
        f"{single_layer_model.num_params(include_embedding=False) + (params_per_block * (n_layers - 1)):,d}"
    )


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(
        yaml_path,
        [clean_opt(s) for s in args_list + ["--data.paths=[]", "--save_folder=/tmp", "--evaluators=[]"]],
        validate_paths=False,
    )
    main(cfg)
