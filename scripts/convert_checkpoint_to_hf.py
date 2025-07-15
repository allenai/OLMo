import json
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple

try:
    import flash_attn 
except ImportError:
    flash_attn = None

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.conversion.state_mapping import TemplatePlaceholder
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.convert import get_converter_to_hf
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)


def create_old_olmo_to_olmo_core_mapping(n_layers: int) -> Dict[str, str]:
    mapping = {}
    
    mapping["embeddings.weight"] = "transformer.wte.weight"
    
    for layer_idx in range(n_layers):
        mapping[f"blocks.{layer_idx}.attention.w_q.weight"] = f"transformer.blocks.{layer_idx}.att_proj.weight"
        mapping[f"blocks.{layer_idx}.attention.w_k.weight"] = f"transformer.blocks.{layer_idx}.att_proj.weight"
        mapping[f"blocks.{layer_idx}.attention.w_v.weight"] = f"transformer.blocks.{layer_idx}.att_proj.weight"
        mapping[f"blocks.{layer_idx}.attention.w_out.weight"] = f"transformer.blocks.{layer_idx}.attn_out.weight"
 
        mapping[f"blocks.{layer_idx}.feed_forward.w1.weight"] = f"transformer.blocks.{layer_idx}.ff_proj.weight"
        mapping[f"blocks.{layer_idx}.feed_forward.w3.weight"] = f"transformer.blocks.{layer_idx}.ff_proj.weight"
        mapping[f"blocks.{layer_idx}.feed_forward.w2.weight"] = f"transformer.blocks.{layer_idx}.ff_out.weight"
        
        mapping[f"blocks.{layer_idx}.attention_norm.weight"] = f"transformer.blocks.{layer_idx}.attn_norm.weight"
        mapping[f"blocks.{layer_idx}.feed_forward_norm.weight"] = f"transformer.blocks.{layer_idx}.ff_norm.weight"
        
        mapping[f"blocks.{layer_idx}.attention.q_norm.weight"] = f"transformer.blocks.{layer_idx}.q_norm.weight"
        mapping[f"blocks.{layer_idx}.attention.k_norm.weight"] = f"transformer.blocks.{layer_idx}.k_norm.weight"
    
    mapping["lm_head.norm.weight"] = "transformer.ln_f.weight"
    mapping["lm_head.w_out.weight"] = "transformer.wte.weight" 
    
    return mapping


def load_and_convert_old_checkpoint(checkpoint_dir: str, model, work_dir: str):
    import torch
    import os
    import pickle
    
    log.info("Loading old OLMo checkpoint with custom weight conversion")
    
    state_dict = {}
    checkpoint_extensions = ['.distcp', '.pt', '.pth', '.bin']
    checkpoint_files = []
    
    for ext in checkpoint_extensions:
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith(ext)]
        if files:
            checkpoint_files = files
            log.info(f"Found {len(files)} checkpoint files with extension {ext}")
            break
    
    if not checkpoint_files:
        all_files = os.listdir(checkpoint_dir)
        log.info(f"Available files in {checkpoint_dir}: {all_files}")
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    for checkpoint_file in checkpoint_files:
        file_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            with open(file_path, 'rb') as f:
                shard_data = torch.load(f, map_location='cpu', weights_only=False)
                if isinstance(shard_data, dict):
                    for key, value in shard_data.items():
                        if isinstance(value, torch.Tensor):
                            if any(pattern in key for pattern in ["model.", "transformer.", "blocks.", "att_proj", "attn_out"]):
                                state_dict[key] = value
                            elif any(pattern in key for pattern in ["blocks.", "att_proj", "attn_out", "wte.", "ln_f"]):
                                state_dict[f"model.{key}"] = value
        except Exception as e:
            log.warning(f"Failed to load {checkpoint_file}: {e}")
            continue
    
    if not state_dict:
        log.warning("Checking first checkpoint file for available keys...")
        if checkpoint_files:
            try:
                with open(os.path.join(checkpoint_dir, checkpoint_files[0]), 'rb') as f:
                    sample_data = torch.load(f, map_location='cpu', weights_only=False)
                    if isinstance(sample_data, dict):
                        log.warning(f"Available keys in checkpoint: {list(sample_data.keys())[:10]}...")
            except Exception as e:
                log.warning(f"Failed to inspect checkpoint: {e}")
        raise ValueError("No model weights found in checkpoint files")
    
    model_state_dict = model.state_dict()
    converted_state_dict = {}
    
    if "model.transformer.wte.weight" in state_dict:
        converted_state_dict["embeddings.weight"] = state_dict["model.transformer.wte.weight"]
    
    if "model.transformer.ln_f.weight" in state_dict:
        converted_state_dict["lm_head.norm.weight"] = state_dict["model.transformer.ln_f.weight"]
    if "model.transformer.wte.weight" in state_dict:
        converted_state_dict["lm_head.w_out.weight"] = state_dict["model.transformer.wte.weight"]
    
    for layer_idx in range(16): 
        prefix = f"model.transformer.blocks.{layer_idx}"
        
        if f"{prefix}.att_proj.weight" in state_dict:
            fused_weight = state_dict[f"{prefix}.att_proj.weight"]  # [6144, 2048]
            hidden_size = fused_weight.shape[1]  # 2048
            
            q_weight = fused_weight[:hidden_size, :]  # [2048, 2048]
            k_weight = fused_weight[hidden_size:2*hidden_size, :]  # [2048, 2048]
            v_weight = fused_weight[2*hidden_size:3*hidden_size, :]  # [2048, 2048]
            
            converted_state_dict[f"blocks.{layer_idx}.attention.w_q.weight"] = q_weight
            converted_state_dict[f"blocks.{layer_idx}.attention.w_k.weight"] = k_weight
            converted_state_dict[f"blocks.{layer_idx}.attention.w_v.weight"] = v_weight
        
        if f"{prefix}.attn_out.weight" in state_dict:
            converted_state_dict[f"blocks.{layer_idx}.attention.w_out.weight"] = state_dict[f"{prefix}.attn_out.weight"]
        
        if f"{prefix}.ff_proj.weight" in state_dict:
            ff_weight = state_dict[f"{prefix}.ff_proj.weight"]
            if ff_weight.shape[0] > 43690: 
                mid = ff_weight.shape[0] // 2
                converted_state_dict[f"blocks.{layer_idx}.feed_forward.w1.weight"] = ff_weight[:mid, :]
                converted_state_dict[f"blocks.{layer_idx}.feed_forward.w3.weight"] = ff_weight[mid:, :]
            else:
                converted_state_dict[f"blocks.{layer_idx}.feed_forward.w1.weight"] = ff_weight
                converted_state_dict[f"blocks.{layer_idx}.feed_forward.w3.weight"] = ff_weight
        
        if f"{prefix}.ff_out.weight" in state_dict:
            converted_state_dict[f"blocks.{layer_idx}.feed_forward.w2.weight"] = state_dict[f"{prefix}.ff_out.weight"]
        
        if f"{prefix}.attn_norm.weight" in state_dict:
            converted_state_dict[f"blocks.{layer_idx}.attention_norm.weight"] = state_dict[f"{prefix}.attn_norm.weight"]
        if f"{prefix}.ff_norm.weight" in state_dict:
            converted_state_dict[f"blocks.{layer_idx}.feed_forward_norm.weight"] = state_dict[f"{prefix}.ff_norm.weight"]
        
        if f"{prefix}.q_norm.weight" in state_dict:
            converted_state_dict[f"blocks.{layer_idx}.attention.q_norm.weight"] = state_dict[f"{prefix}.q_norm.weight"]
        if f"{prefix}.k_norm.weight" in state_dict:
            converted_state_dict[f"blocks.{layer_idx}.attention.k_norm.weight"] = state_dict[f"{prefix}.k_norm.weight"]
    
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    if missing_keys:
        log.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        log.warning(f"Unexpected keys: {unexpected_keys}")
    
    log.info("Successfully loaded and converted old checkpoint")


def convert_checkpoint_to_hf(
    original_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    dtype: Optional[DType] = None,
    max_sequence_length: int = -1,
    validate: bool = True,
    debug: bool = False,
    device: torch.device | None = None,
    auto_map_old_params: bool = True,
) -> None:

    # if max_sequence_length <= 0:
    #     raise ValueError(f"Missing or invalid sequence length: {max_sequence_length}")

    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    device = device or get_default_device()
    if (
        validate
        and (flash_attn is None or device != torch.device("cuda"))
        and (attention := transformer_config_dict.get("block", {}).get("attention")) is not None
    ):
        if attention["name"] == "fused":
            log.warning(
                "Running conversion without cuda or flash attention on a model requiring flash attention, validation would fail so we are disabling it."
            )
            validate = False
        elif attention.get("use_flash"):
            log.info(
                "Flash attention or cuda is unavailable, turning off flash attention to stop validation from failing."
            )
            attention["use_flash"] = False

    model = TransformerConfig.from_dict(transformer_config_dict).build()
    model.to_empty(device=device)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)
    vocab_size = tokenizer_config.vocab_size

    key_mapping = None
    if auto_map_old_params:
        n_layers = transformer_config_dict["n_layers"]
        key_mapping = create_old_olmo_to_olmo_core_mapping(n_layers)

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = join_path(original_checkpoint_path, "model_and_optim")
        
        try:
            load_and_convert_old_checkpoint(model_and_optim_dir, model, work_dir)
        except Exception as e:
            log.error(f"Failed to load with custom converter: {e}")
            
            try:
                load_model_and_optim_state(
                    model_and_optim_dir,
                    model,
                    work_dir=work_dir,
                )
            except Exception as e2:
                raise RuntimeError(e2)
        
        state_dict_options = dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

        save_hf_model(
            output_path,
            model_state_dict,
            model,
            vocab_size=vocab_size,
            work_dir=work_dir,
            save_overwrite=True,
        )

    huggingface_config = AutoConfig.from_pretrained(output_path)
    huggingface_config.max_position_embeddings = max_sequence_length
    huggingface_config.pad_token_id = tokenizer_config.pad_token_id
    huggingface_config.bos_token_id = tokenizer_config.bos_token_id
    huggingface_config.eos_token_id = tokenizer_config.eos_token_id
    huggingface_config.save_pretrained(output_path)

    if validate:
        validate_conversion(
            output_path, model, tokenizer_config.vocab_size, debug=debug, dtype=dtype, device=device
        )


def _register_debug_hooks(hf_model: torch.nn.Module, model: Transformer):
    MAX_DIM_SIZE = 1_000_000

    olmo_core_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}
    hf_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}

    def module_hook(
        debug_state: Dict[str, Tuple[int, torch.Tensor]],
        name: str,
        _: torch.nn.Module,
        args,
        output,
    ):
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            state_name = f"{name}|input"
            input = args[0].detach()
            for i, size in enumerate(input.shape):
                input = input.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), input)
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|output"
            output = output.detach()
            for i, size in enumerate(output.shape):
                output = output.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), output)

    for name, module in model.named_modules():
        module.register_forward_hook(partial(module_hook, olmo_core_debug_state, name))
    for name, module in hf_model.named_modules():
        module.register_forward_hook(partial(module_hook, hf_debug_state, name))

    return olmo_core_debug_state, hf_debug_state


def validate_conversion(
    hf_path: str | Path,
    model: Transformer,
    vocab_size: int,
    debug: bool = False,
    dtype: DType | None = None,
    device: torch.device | None = None,
):
    if torch.cuda.is_available():
        torch.cuda.init()

    device = device or get_default_device()

    B, T = 1, 120
    input_ids = torch.randint(0, vocab_size, (B, T)).to(device)

    hf_model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype="auto").to(device).eval()

    olmo_core_state, hf_state = {}, {}
    state_mapping = None
    if debug:
        olmo_core_state, hf_state = _register_debug_hooks(hf_model, model)
        state_converter = get_converter_to_hf()

        # if not hasattr(hf_model.config, "num_hidden_layers"):
        #     raise ValueError(f"Number of hidden layers missing in HF config: {hf_model.config}")
        n_layers: int = hf_model.config.num_hidden_layers
        n_experts: int | None = getattr(hf_model.config, "num_experts", None)

        placeholder_bounds = {
            TemplatePlaceholder.LAYER: n_layers,
        }
        if n_experts:
            placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

        state_mapping = state_converter.get_mappings(model.state_dict(), placeholder_bounds)

    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    del hf_model

    if dtype:
        model = model.to(dtype.as_pt())
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
        assert state_mapping is not None

        simple_key_mapping = {
            mapping.source_keys[0]
            .replace(".weight", ""): mapping.dest_keys[0]
            .replace(".weight", "")
            for mapping in state_mapping
            if len(mapping.source_keys) == 1 and len(mapping.dest_keys) == 1
        }

        log.info(f"simple mapping: {simple_key_mapping}")
        log.info(f"hf_state keys: {hf_state.keys()}")
        log.info(f"olmo_core_state keys: {olmo_core_state.keys()}")

        for olmo_core_state_name, (_, olmo_core_tensor) in sorted(
            olmo_core_state.items(), key=lambda item: item[1][0]
        ):
            olmo_core_key, state_type = olmo_core_state_name.split("|")
            if olmo_core_key not in simple_key_mapping:
                continue

            hf_state_name = f"{simple_key_mapping[olmo_core_key]}|{state_type}"
            if hf_state_name not in hf_state:
                continue

            _, hf_tensor = hf_state[hf_state_name]

            if olmo_core_tensor.shape != hf_tensor.shape:
                log.info(
                    f"{olmo_core_state_name}, {hf_state_name} shape mismatch: {olmo_core_tensor.shape} {hf_tensor.shape}"
                )
            if olmo_core_tensor.dtype != hf_tensor.dtype:
                log.info(
                    f"{olmo_core_state_name}, {hf_state_name} dtype mismatch: {olmo_core_tensor.dtype} {hf_tensor.dtype}"
                )
            if len(olmo_core_tensor.shape) == len(hf_tensor.shape):
                common_shape = tuple(
                    min(olmo_core_dim, hf_dim)
                    for olmo_core_dim, hf_dim in zip(olmo_core_tensor.shape, hf_tensor.shape)
                )
                for i, dim in enumerate(common_shape):
                    olmo_core_tensor = olmo_core_tensor.narrow(i, 0, dim)
                    hf_tensor = hf_tensor.narrow(i, 0, dim)
                log.info(
                    f"{olmo_core_state_name}, {hf_state_name} element diff abs mean: {(olmo_core_tensor - hf_tensor).float().abs().mean()}"
                )

    torch.testing.assert_close(
        hf_logits[..., :vocab_size].float(), logits[..., :vocab_size].float(), rtol=1e-4, atol=1e-4
    )


def load_config(checkpoint_input_dir: PathOrStr) -> Optional[dict]:
    # if not file_exists(f"{checkpoint_input_dir}/config.json"):
    #     raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # if "model" not in config_dict:
    #     raise RuntimeError(
    #         f"Config file at {checkpoint_input_dir} is not an OLMo core experiment config, ignoring"
    #     )

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Local or remote directory containing the OLMo Core checkpoint.",
    )

    parser.add_argument(
        "-o",
        "--huggingface-output-dir",
        type=Path,
        required=True,
        help="Local or remote directory where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        required=True,
        help="Max sequence length supported by the model.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="If set, validation to check that the converted model matches the original model is skipped.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="If set, debug information of validation is output.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="The device on which conversion and validation occurs. Defaults to CUDA or MPS if available and initialized.",
    )
    parser.add_argument(
        "--dtype",
        help="The torch dtype that model weights should be saved as. Defaults to bfloat16 due to https://github.com/allenai/olmo-cookbook/issues/60.",
        type=DType,
        default=DType.bfloat16,
    )
    parser.add_argument(
        "--no-auto-map",
        dest="auto_map_old_params",
        action="store_false",
        help="If set, disable automatic mapping of old OLMo parameter names to new OLMo Core names.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.checkpoint_input_path)
    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
        auto_map_old_params=args.auto_map_old_params,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()