import torch

from olmo.config import EvaluatorType, TrainConfig
from olmo.eval import ICLMetric, build_downstream_evaluator
from olmo.eval.evaluator import Evaluator


def test_icl_metric_reports_skipped_documents():
    metric = ICLMetric()
    metric.loglikelihoods = [
        torch.tensor((0, 0, 2.0)),
        torch.tensor((0, 1, 1.0)),
        torch.tensor((1, 0, 0.0)),
        torch.tensor((1, 1, 3.0)),
        # Document 2 is missing continuation 1 but contains continuation 2.
        torch.tensor((2, 0, 5.0)),
        torch.tensor((2, 2, 4.0)),
    ]
    metric.labels = [
        torch.tensor((0, 0, 0)),
        torch.tensor((0, 1, 0)),
        torch.tensor((1, 0, 0)),
        torch.tensor((1, 1, 0)),
        torch.tensor((2, 0, 0)),
        torch.tensor((2, 2, 0)),
    ]
    evaluator = Evaluator(
        label="synthetic",
        type=EvaluatorType.downstream,
        eval_loader=None,
        eval_metric=metric,
    )

    assert evaluator.compute_metrics() == {
        "eval/downstream/synthetic_acc": 0.5,
        "eval/downstream/synthetic_skipped_docs": 1,
    }


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
