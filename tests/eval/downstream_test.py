import torch

from olmo.config import TrainConfig
from olmo.eval import build_downstream_evaluator


def test_piqa():
    cfg = TrainConfig.load("test_fixtures/train_tiny_with_evaluator.yaml")
    from olmo.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_train_config(cfg)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    evaluator = build_downstream_evaluator(
        cfg, cfg.evaluators[1], tokenizer, torch.device("cpu"), is_unit_test=True
    )
    logits = torch.rand(4, 57, 50304)
    first_batch = next(iter(evaluator.eval_loader))
    evaluator.reset_metrics()
    evaluator.update_metrics(first_batch, logits.sum(), logits)
