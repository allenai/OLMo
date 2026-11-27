import torch

from olmo.config import TrainConfig
from olmo.eval import build_downstream_evaluator
from olmo.eval.downstream import ICLMetric


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


def test_icl_metric_empty():
    metric = ICLMetric(metric_type="acc")
    assert metric.compute().item() == 0.0

    metric_f1 = ICLMetric(metric_type="f1")
    assert metric_f1.compute().item() == 0.0


def test_icl_metric_incomplete_continuations():
    metric = ICLMetric(metric_type="acc")
    # doc 0 has complete continuations (0 and 1)
    # doc 1 has incomplete continuations (0 and 2, missing 1)
    # doc 2 has incomplete continuations (0 only, but label is 2)
    metric.labels = [
        torch.LongTensor([0, 0, 0]),
        torch.LongTensor([0, 1, 0]),
        torch.LongTensor([1, 0, 0]),
        torch.LongTensor([1, 2, 0]),
        torch.LongTensor([2, 0, 2]),
    ]
    metric.loglikelihoods = [
        torch.Tensor([0.0, 0.0, -1.0]),
        torch.Tensor([0.0, 1.0, -2.0]),
        torch.Tensor([1.0, 0.0, -1.0]),
        torch.Tensor([1.0, 2.0, -2.0]),
        torch.Tensor([2.0, 0.0, -1.0]),
    ]
    # doc 0 is correct (argmax is 0 == label 0)
    # doc 1 is skipped (cont 1 missing)
    # doc 2 is skipped (label is 2, cont 1 and 2 missing)
    result = metric.compute()
    assert result.item() == 1.0
    assert metric.num_skipped_docs == 2
