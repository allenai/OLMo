from itertools import islice
import json
import os
import sys
from tqdm import tqdm
from typing import Any, Dict
import torch
import torch.nn.functional as F
import transformers
from olmo.config import TrainConfig, EvaluatorConfig, EvaluatorType
from olmo.eval import build_evaluator
from olmo.torch_util import move_to_device
from olmo.exceptions import OLMoCliError


def get_labels(batch: Dict[str, Any]) -> torch.Tensor:
    # Labels are just input IDs shifted to the left (first item is ignored).
    labels, label_mask, attention_mask, instance_mask = (
        batch["input_ids"].clone(),
        batch.get("label_mask"),
        batch.get("attention_mask"),
        batch.get("instance_mask"),
    )
    if label_mask is not None:
        labels.masked_fill_(~label_mask, -100)
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0.0, -100)
    if instance_mask is not None:
        labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
    return labels[..., 1:].contiguous()

def main(cfg: TrainConfig, model_name: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    cfg.device_eval_batch_size = 4
    cfg.evaluators = [
        EvaluatorConfig(label="piqa_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="piqa_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="piqa_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="piqa_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="piqa_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="piqa_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="hellaswag_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="hellaswag_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="hellaswag_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="hellaswag_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="hellaswag_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="hellaswag_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="winogrande_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="winogrande_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="winogrande_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="winogrande_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="winogrande_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="winogrande_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="openbookqa_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="openbookqa_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="openbookqa_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="openbookqa_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="openbookqa_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="openbookqa_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="boolq_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="boolq_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="boolq_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="boolq_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="boolq_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="boolq_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="sciq_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="sciq_rc_0shot_bpb", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="sciq_rc_5shot", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="sciq_rc_5shot_bpb", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="sciq_mc_5shot", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="sciq_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_easy_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_easy_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_easy_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_easy_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_easy_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_easy_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_challenge_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_challenge_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_challenge_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_challenge_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_challenge_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="arc_challenge_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="copa_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="copa_rc_0shot_bpb", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="copa_rc_5shot", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="copa_rc_5shot_bpb", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="copa_mc_5shot", type=EvaluatorType.downstream),
        # EvaluatorConfig(label="copa_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="csqa_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="csqa_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="csqa_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="csqa_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="csqa_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="csqa_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="socialiqa_rc_0shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="socialiqa_rc_0shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="socialiqa_rc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="socialiqa_rc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="socialiqa_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="socialiqa_mc_5shot_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_stem_var", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_humanities_var", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_social_sciences_var", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_other_var", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_stem_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_humanities_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_social_sciences_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_other_mc_5shot", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_stem_var_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_humanities_var_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_social_sciences_var_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_other_var_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_stem_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_humanities_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_social_sciences_bpb", type=EvaluatorType.downstream),
        EvaluatorConfig(label="mmlu_other_bpb", type=EvaluatorType.downstream),
    ]

    evaluators = []
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, tokenizer, device))

    eval_metrics = {}
    for evaluator in tqdm(evaluators):
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
        for eval_step, eval_batch in enumerate(eval_batches):
            batch = move_to_device(eval_batch, device)
            with torch.no_grad():
                with torch.autocast("cuda", enabled=True, dtype=cfg.autocast_precision):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                    ).logits
                    logits_for_loss = logits[..., :-1, :].contiguous()
                    # shape: (batch_size * seq_len, vocab_size)
                    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
                    # shape: (batch_size, seq_len)
                    labels = get_labels(batch)
                    # shape: (batch_size * seq_len,)
                    labels = labels.view(-1)
                    ce_loss = F.cross_entropy(logits_for_loss, labels, ignore_index=-100, reduction="none")
                    # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
                    ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
                ce_loss = ce_loss.mean(dim=-1)
            evaluator.update_metrics(batch, ce_loss, logits)

        # Get final metrics.
        metrics = evaluator.compute_metrics()
        eval_metrics.update(metrics)
        print(metrics)

        del eval_batches

    print(eval_metrics)

    save_folder = f'/weka/oe-training-default/jiachengl/hc-law/eval_bpb_mc'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f'{save_folder}/{model_name.replace("/", "_")}.json', 'w') as f:
        json.dump(eval_metrics, f)


if __name__ == "__main__":

    try:
        yaml_path, model_name = sys.argv[1], sys.argv[2]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [MODEL_NAME]")

    cfg = TrainConfig.load(yaml_path)
    main(cfg, model_name)
