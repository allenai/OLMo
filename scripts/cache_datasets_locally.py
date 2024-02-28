import sys
from olmo.eval import build_downstream_evaluator
from olmo.exceptions import OlmoCliError
from olmo.config import TrainConfig
from olmo.tokenizer import Tokenizer
import torch


def clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


def main(cfg: TrainConfig):
    tokenizer = Tokenizer.from_train_config(cfg)

    for eval_cfg in cfg.evaluators:
        if eval_cfg.type != 'downstream':
            continue
        evaluator = build_downstream_evaluator(
            cfg, eval_cfg, tokenizer, torch.device("cpu"), is_unit_test=True
        )
        print(evaluator)


if __name__ == "__main__":
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
