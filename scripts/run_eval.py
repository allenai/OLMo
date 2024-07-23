import argparse
import logging
from itertools import islice
from typing import Any, Dict, List

import torch
from cached_path import cached_path

from olmo import Tokenizer
from olmo.config import EvaluatorConfig, EvaluatorType, TrainConfig
from olmo.eval import Evaluator, build_downstream_evaluator
from olmo.model import OLMo
from olmo.torch_util import (
    barrier,
    get_local_rank,
    move_to_device,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import prepare_cli_environment

log = logging.getLogger("eval")


def eval_step(
    cfg: TrainConfig, model: OLMo, batch: Dict[str, Any], evaluator: Evaluator, device: torch.device
) -> None:
    # Move tensors to the right device.
    batch = move_to_device(batch, device)

    # Run forward pass.
    with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
        with torch.autocast("cuda", enabled=True, dtype=cfg.autocast_precision):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                attention_bias=batch.get("attention_bias"),
                doc_lens=batch.get("doc_lens"),
                max_doc_lens=batch.get("max_doc_lens"),
            ).logits

    # Update metrics. Only downstream, we don't care about lm type ce loss here; setting to 0.0
    evaluator.update_metrics(
        batch, torch.Tensor([0.0]), logits
    )  # batch includes all keys that the downstream evaluation needs

    barrier()


def main(checkpoint_path: str, eval_configs: List[EvaluatorConfig]):
    yaml_path = checkpoint_path + "/" + "config.yaml"
    cfg = TrainConfig.load(cached_path(yaml_path))

    prepare_cli_environment()
    log.info("CLI environment prepared")

    # Set seed.
    seed_all(cfg.seed)

    # Set CUDA device.

    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info("Building model...")
    olmo_model = OLMo.from_checkpoint(checkpoint_path).to(device).eval()
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

    tokenizer = Tokenizer.from_train_config(cfg)

    evaluators = [build_downstream_evaluator(cfg, eval_config, tokenizer, device) for eval_config in eval_configs]

    # TODO: temporary, remove
    cfg = cfg.update_with(eval_subset_num_batches=10)

    eval_metrics = {}
    for evaluator in evaluators:
        log.info(f"Running evaluation for '{evaluator.label}'...")

        # Reset metrics.
        evaluator.reset_metrics()

        # Initialize data loader iterator.
        eval_batches = iter(evaluator.eval_loader)

        # Adjust how many batches to evaluate on.
        num_eval_batches = (
            evaluator.subset_num_batches
            if evaluator.subset_num_batches is not None
            else cfg.eval_subset_num_batches
        )
        if num_eval_batches > 0:
            num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
            eval_batches = islice(eval_batches, num_eval_batches)

        # Run model over batches.
        for eval_step_n, eval_batch in enumerate(eval_batches):
            eval_step(cfg, olmo_model, eval_batch, evaluator, device)

            # Log to console.
            if eval_step_n + 1 == num_eval_batches or (eval_step_n + 1) % cfg.console_log_interval == 0:
                log.info(f"[eval_step={eval_step_n + 1}/{num_eval_batches}]")

        # Get final metrics.
        metrics = evaluator.compute_metrics()
        eval_metrics.update(metrics)
        Trainer.log_metrics_to_console(f"{evaluator.label}", metrics)

        del eval_batches

    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run specified downstream evals for a checkpoint",
    )

    parser.add_argument("checkpoint_path")

    parser.add_argument("-e", "--downstream-evals", nargs="+", default=[])

    args = parser.parse_args()
    print(args)

    eval_configs = [
        EvaluatorConfig(label=eval_name, type=EvaluatorType.downstream) for eval_name in args.downstream_evals
    ]

    main(args.checkpoint_path, eval_configs)
