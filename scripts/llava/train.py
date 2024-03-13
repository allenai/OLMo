"""Wrapper for LLaVA Training"""
import os
from dataclasses import dataclass, field, asdict
import logging
from pathlib import Path
from typing import Dict, Optional, List
import wandb

import torch

import transformers

from olmo.config import TrainConfig, ModelConfig, DataConfig
from olmo.data import DataCollator
from olmo.torch_util import barrier
from hf_olmo import *


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="allenai/OLMo-7B-Instruct")
    cache_dir: Optional[str] = field(default=None)
    tune_mm_adapter: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    config_file: str = field(default=None, metadata={"help": "Path to the OlMo train configuration file"})
    optim: str = field(default="adamw_torch")
    seed: int = field(default=6198, metadata={"help": "Seed for training"})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_adapter", False):
        # Only save Adapter
        keys_to_match = ['image_newline', 'projector', 'resampler']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_adapter_folder = os.path.join(parent_folder, "mm_adapter")
                os.makedirs(mm_adapter_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_adapter_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_adapter.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, model_cfg: ModelConfig, data_cfg: DataConfig) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = build_train_dataset(tokenizer, model_cfg, data_cfg)
    data_collator = DataCollator(data_cfg.pad_direction, model_cfg.pad_token_id)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    olmo_train_cfg = TrainConfig.load(training_args.config_file)
    olmo_train_cfg.model.precision = olmo_train_cfg.precision
    olmo_model_cfg, olmo_data_cfg = olmo_train_cfg.model, olmo_train_cfg.data

    local_rank = training_args.local_rank
    compute_dtype = olmo_train_cfg.autocast_precision

    model = OLMoForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=(torch.bfloat16 if compute_dtype == torch.bfloat16 else None),
    )
    model.config.use_cache = False

    """
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    """
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=olmo_model_cfg.max_sequence_length,
    )

    if model.get_vision_backbone() is None:
        model.get_model().initialize_vision_backbone(olmo_model_cfg)
    model.get_vision_backbone().to(
        dtype=torch.bfloat16 if compute_dtype == torch.bfloat16 else torch.float16,
        device=training_args.device,
    )
    model.config.tune_mm_adapter = training_args.tune_mm_adapter = model_args.tune_mm_adapter
    if model_args.tune_mm_adapter:
        model.get_language_model().requires_grad_(False)
    data_module = make_supervised_data_module(tokenizer, olmo_model_cfg, olmo_data_cfg)

    if isinstance(training_args.report_to, List):
        report_to_wandb = "wandb" in training_args.report_to
    else:
        report_to_wandb = training_args.report_to == "wandb"
    
    if report_to_wandb and olmo_train_cfg.wandb and (training_args.process_index == 0 or not olmo_train_cfg.wandb.rank_zero_only):
        wandb_dir = Path(training_args.output_dir) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_config = {'training': asdict(training_args), 'model': olmo_model_cfg.asdict(), 'data': olmo_data_cfg.asdict()}
        wandb.init(
            dir=wandb_dir,
            project=olmo_train_cfg.wandb.project,
            entity=olmo_train_cfg.wandb.entity,
            group=olmo_train_cfg.wandb.group,
            name=olmo_train_cfg.wandb.name,
            tags=olmo_train_cfg.wandb.tags,
            config=wandb_config,
        )

    barrier()

    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()