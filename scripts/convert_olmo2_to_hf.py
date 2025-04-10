import argparse
import base64
import gc
import glob
import json
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

"""
Sample Usage:

1. Download the unsharded checkpoints from R2.

2. To convert multiple checkpoints to Hugging Face format, run:
```
python src/transformers/models/olmo2/convert_olmo2_weights_to_hf.py \
    --input_base_dir /path/to/checkpoints/base/dir \
    --output_base_dir /output/base/path \
    --folder_name olmo-13b-1124_stage2_ingredient1
```

3. To convert a single checkpoint to Hugging Face format, run:
```
python src/transformers/models/olmo2/convert_olmo2_weights_to_hf.py \
    --input_base_dir /path/to/checkpoints/base/dir \
    --output_base_dir /output/base/path \
    --single_checkpoint olmo-13b-1124_stage2_ingredient1/step10000-unsharded
```

Thereafter, models can be loaded via:

```py
from transformers import Olmo2ForCausalLM, AutoTokenizer

model = Olmo2ForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```
"""


def decode_key(encoded_key):
    """Decode a base64-encoded pickled key."""
    try:
        decoded = base64.b64decode(encoded_key)
        unpickled = pickle.loads(decoded)
        return unpickled
    except:
        return encoded_key


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(
    model_path,
    input_base_path,
    include_tokenizer=True,
    tokenizer_path=None,
    safe_serialization=True,
    fix_eos_token_id=True,
    tmp_cleanup=True,
):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.yaml"
    olmo2_config = yaml.safe_load(config_path.read_text())["model"]

    if not olmo2_config.get("attention_layer_norm", False):
        raise RuntimeError("OLMo2 checkpoints must have attention layer norm")
    if not olmo2_config.get("norm_after", False):
        raise RuntimeError("OLMo2 checkpoints must set norm_after to True")

    n_layers = olmo2_config["n_layers"]
    n_heads = olmo2_config["n_heads"]
    dim = olmo2_config["d_model"]
    dims_per_head = dim // n_heads
    base = olmo2_config["rope_theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = olmo2_config["max_sequence_length"]

    vocab_size = olmo2_config.get("embedding_size", olmo2_config["vocab_size"])

    if olmo2_config.get("n_kv_heads", None) is not None:
        num_key_value_heads = olmo2_config["n_kv_heads"]  # for GQA / MQA
    elif olmo2_config["multi_query_attention"]:  # compatibility with other checkpoints
        num_key_value_heads = 1
    else:
        num_key_value_heads = n_heads

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

    # Not sharded
    # (The sharded implementation would also work, but this is simpler.)
    encoded_state_dict = load_file(os.path.join(input_base_path, "model.safetensors"), device="cpu")

    loaded = {}
    for encoded_key, value in encoded_state_dict.items():
        decoded_key = decode_key(encoded_key)

        # Handle the tuple format (('key',), False)
        if (
            isinstance(decoded_key, tuple)
            and len(decoded_key) == 2
            and isinstance(decoded_key[0], tuple)
            and len(decoded_key[0]) == 1
        ):
            actual_key = decoded_key[0][0]
            loaded[actual_key] = value
        else:
            loaded[decoded_key] = value

    del encoded_state_dict
    gc.collect()

    param_count = 0
    index_dict: Dict[str, Any] = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        # Unsharded
        # TODO: Layernorm stuff
        # TODO: multi query attention
        fused_dims = [dim, dims_per_head * num_key_value_heads, dims_per_head * num_key_value_heads]
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            loaded[f"transformer.blocks.{layer_i}.att_proj.weight"], fused_dims, dim=0
        )
        up_proj_weight, gate_proj_weight = torch.chunk(
            loaded[f"transformer.blocks.{layer_i}.ff_proj.weight"], 2, dim=0
        )
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight": v_proj_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                f"transformer.blocks.{layer_i}.attn_out.weight"
            ],
            f"model.layers.{layer_i}.self_attn.q_norm.weight": loaded[
                f"transformer.blocks.{layer_i}.q_norm.weight"
            ],
            f"model.layers.{layer_i}.self_attn.k_norm.weight": loaded[
                f"transformer.blocks.{layer_i}.k_norm.weight"
            ],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": gate_proj_weight,
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"transformer.blocks.{layer_i}.ff_out.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": up_proj_weight,
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                f"transformer.blocks.{layer_i}.attn_norm.weight"
            ],
            f"model.layers.{layer_i}.post_feedforward_layernorm.weight": loaded[
                f"transformer.blocks.{layer_i}.ff_norm.weight"
            ],
        }

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"

    # Unsharded
    # TODO: Deal with weight-tying
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "model.norm.weight": loaded["transformer.ln_f.weight"],
        "lm_head.weight": (
            loaded["transformer.ff_out.weight"]
            if "transformer.ff_out.weight" in loaded
            else loaded["transformer.wte.weight"]
        ),
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    if olmo2_config.get("mlp_hidden_size", None) is not None:
        intermediate_size = olmo2_config["mlp_hidden_size"] // 2
    else:
        intermediate_size = (dim * olmo2_config["mlp_ratio"]) // 2

    if fix_eos_token_id and olmo2_config["eos_token_id"] == 0:
        # Fixing a bug in OLMo where eos token id was incorrectly set
        print("Changing eos_token_id from 0 to 50279.")
        olmo2_config["eos_token_id"] = 50279

    config = Olmo2Config(
        vocab_size=vocab_size,
        hidden_size=dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=olmo2_config["pad_token_id"],
        bos_token_id=None,
        eos_token_id=olmo2_config["eos_token_id"],
        tie_word_embeddings=olmo2_config["weight_tying"],
        rms_norm_eps=olmo2_config["layer_norm_eps"],
        rope_theta=base,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        _write_tokenizer(model_path, config, input_base_path, tokenizer_path)

    print(f"Loading the checkpoint in a OLMo2 model from {tmp_model_path}.")
    model = Olmo2ForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print(f"Saving in the Transformers format to {model_path}.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    if tmp_cleanup:
        # Make cleanup optional; attempting to `rmtree` the `tmp_model_path` causes
        # errors if using NFS.
        shutil.rmtree(tmp_model_path)

    print(f"Successfully converted {input_base_path} to {model_path}")


def _write_tokenizer(
    output_path: Path,
    config: Olmo2Config,
    checkpoint_dir: str,
    input_tokenizer_path: Path | None,
) -> None:
    print(f"Saving a {GPT2TokenizerFast.__name__} to {output_path}.")

    if input_tokenizer_path is not None:
        base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))
    else:
        config_path = Path(checkpoint_dir) / "config.yaml"
        tokenizer_config = yaml.safe_load(config_path.read_text())["tokenizer"]

        # Initialize tokenizer and validate vocab size.
        if Path(tokenizer_config["identifier"]).is_file():
            base_tokenizer = Tokenizer.from_file(tokenizer_config["identifier"])
        else:
            base_tokenizer = Tokenizer.from_pretrained(tokenizer_config["identifier"])

    eos_token_id = config.eos_token_id if config.eos_token_id is not None else base_tokenizer.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else eos_token_id

    tokenizer = GPT2TokenizerFast(
        tokenizer_object=base_tokenizer,
        eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
    )

    tokenizer.save_pretrained(output_path)


def get_step_number(checkpoint_path):
    """Extract step number from checkpoint path."""
    match = re.search(r"step(\d+)", os.path.basename(checkpoint_path))
    if match:
        return match.group(1)
    return "unknown"


def process_all_checkpoints(
    input_base_dir,
    output_base_dir,
    folder_name,
    include_tokenizer=True,
    tokenizer_path=None,
    safe_serialization=True,
    fix_eos_token_id=True,
    tmp_cleanup=True,
):
    checkpoint_pattern = os.path.join(input_base_dir, folder_name, "step*-unsharded")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        print(f"No checkpoints found matching pattern: {checkpoint_pattern}")
        return

    checkpoints.sort(key=lambda x: int(get_step_number(x)))

    print(f"Found {len(checkpoints)} checkpoints to process")
    os.makedirs(output_base_dir, exist_ok=True)

    for checkpoint in checkpoints:
        step_number = get_step_number(checkpoint)
        output_dir = os.path.join(output_base_dir, f"{folder_name}_hf", f"step{step_number}_hf")

        print(f"\n\nProcessing checkpoint: {checkpoint}")
        print(f"Output directory: {output_dir}")

        write_model(
            model_path=output_dir,
            input_base_path=checkpoint,
            include_tokenizer=include_tokenizer,
            tokenizer_path=tokenizer_path,
            safe_serialization=safe_serialization,
            fix_eos_token_id=fix_eos_token_id,
            tmp_cleanup=tmp_cleanup,
        )

        print(f"Completed conversion for step{step_number}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_base_dir",
        required=True,
        help="Base directory containing checkpoint directories (e.g., /data/input/amanr/checkpoints13b_stage2/)",
    )
    parser.add_argument(
        "--folder_name",
        required=True,
        help="Model name (e.g., olmo-13b-1124_stage2_ingredient1)",
    )
    parser.add_argument(
        "--output_base_dir",
        required=True,
        help="Base directory where HF models will be written (e.g., /data/input/amanr/)",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not convert OLMo tokenizer to HF tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        type=Path,
        default=None,
        help="Location of OLMo2 tokenizer json file. Defaults to what is set in the config file.",
    )
    parser.add_argument(
        "--no_fix_eos_token_id",
        action="store_false",
        dest="fix_eos_token_id",
        help="If set, does not change eos token id from 0 to 50279 if it is 0. Changing 0 to 50279 is a bug fix, so use this option with care.",
    )
    parser.add_argument(
        "--no_tmp_cleanup",
        action="store_false",
        dest="tmp_cleanup",
        help="If passed, don't remove temp dir at end of HF conversion.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_false",
        dest="safe_serialization",
        help="Whether or not to save using `safetensors`.",
    )
    parser.add_argument(
        "--single_checkpoint",
        default=None,
        help="Process only a specific step checkpoint (e.g., step1000-unsharded). If not provided, all checkpoints will be processed.",
    )

    args = parser.parse_args()

    if args.single_checkpoint:
        checkpoint_path = os.path.join(args.input_base_dir, args.folder_name, args.single_checkpoint)
        step_number = get_step_number(args.single_checkpoint)
        output_dir = os.path.join(args.output_base_dir, f"{args.folder_name}_hf", f"step{step_number}_hf")

        print(f"Processing single checkpoint: {checkpoint_path}")
        print(f"Output directory: {output_dir}")

        write_model(
            model_path=output_dir,
            input_base_path=checkpoint_path,
            include_tokenizer=args.include_tokenizer,
            tokenizer_path=args.tokenizer_json_path,
            safe_serialization=args.safe_serialization,
            fix_eos_token_id=args.fix_eos_token_id,
            tmp_cleanup=args.tmp_cleanup,
        )
    else:
        process_all_checkpoints(
            input_base_dir=args.input_base_dir,
            output_base_dir=args.output_base_dir,
            folder_name=args.folder_name,
            include_tokenizer=args.include_tokenizer,
            tokenizer_path=args.tokenizer_json_path,
            safe_serialization=args.safe_serialization,
            fix_eos_token_id=args.fix_eos_token_id,
            tmp_cleanup=args.tmp_cleanup,
        )


if __name__ == "__main__":
    main()
