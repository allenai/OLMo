import argparse
import json
from typing import Dict, Any

import yaml


def map_layer_norm_type(layer_norm_type: str) -> str:
    norm_map = {
        "rms": "rms",
        "default": "default",
        "layer_norm": "default"
    }
    return norm_map.get(layer_norm_type, "rms")


def map_block_type(block_type: str, norm_after: bool = True) -> str:
    if block_type == "sequential" and norm_after:
        return "reordered_norm"
    if block_type == "sequential" and not norm_after:
        return "default"
    return "default"


def convert_model_config(yaml_model: Dict[str, Any]) -> Dict[str, Any]:
    mlp_hidden_size = yaml_model.get("mlp_hidden_size")
    if mlp_hidden_size is None:
        mlp_ratio = yaml_model.get("mlp_ratio", 8)
        d_model = yaml_model["d_model"]
        mlp_hidden_size = int(1.5 * 8 * d_model / 3) 

    rope_config = None
    if yaml_model.get("rope", True):
        rope_config = {
            "name": "default",
            "theta": yaml_model.get("rope_theta", 500000),
            "full_precision": yaml_model.get("rope_full_precision", True),
            "_CLASS_": "olmo_core.nn.rope.RoPEConfig"
        }

    attention_config = {
        "name": "default",
        "n_heads": yaml_model["n_heads"],
        "n_kv_heads": yaml_model.get("n_kv_heads"),
        "bias": yaml_model.get("include_bias", False),
        "use_flash": yaml_model.get("flash_attention", False),
        "dtype": "float32",
        "_CLASS_": "olmo_core.nn.attention.AttentionConfig"
    }

    if rope_config:
        attention_config["rope"] = rope_config

    if yaml_model.get("attention_layer_norm", True):
        attention_config["qk_norm"] = {
            "name": map_layer_norm_type(yaml_model.get("layer_norm_type", "rms")),
            "eps": yaml_model.get("layer_norm_eps", 1e-6),
            "bias": yaml_model.get("bias_for_layer_norm", False),
            "dtype": "float32",
            "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig"
        }

    layer_norm_config = {
        "name": map_layer_norm_type(yaml_model.get("layer_norm_type", "rms")),
        "eps": yaml_model.get("layer_norm_eps", 1e-6),
        "bias": yaml_model.get("bias_for_layer_norm", False),
        "dtype": "float32",
        "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig"
    }

    feed_forward_config = {
        "hidden_size": mlp_hidden_size,
        "name": "default",
        "bias": yaml_model.get("include_bias", False),
        "dtype": "float32",
        "_CLASS_": "olmo_core.nn.feed_forward.FeedForwardConfig"
    }

    block_config = {
        "attention": attention_config,
        "layer_norm": layer_norm_config,
        "feed_forward": feed_forward_config,
        "name": map_block_type(
            yaml_model.get("block_type", "sequential"),
            yaml_model.get("norm_after", True)
        ),
        "_CLASS_": "olmo_core.nn.transformer.config.TransformerBlockConfig"
    }

    lm_head_config = {
        "name": "default",
        "layer_norm": layer_norm_config,
        "bias": yaml_model.get("include_bias", False),
        "dtype": "float32",
        "_CLASS_": "olmo_core.nn.lm_head.LMHeadConfig"
    }

    transformer_config = {
        "d_model": yaml_model["d_model"],
        "vocab_size": yaml_model.get("embedding_size", yaml_model["vocab_size"]),
        "n_layers": yaml_model["n_layers"],
        "block": block_config,
        "lm_head": lm_head_config,
        "name": "default",
        "dtype": "float32",
        "init_method": "normal",
        "init_seed": 0,
        "_CLASS_": "olmo_core.nn.transformer.config.TransformerConfig"
    }

    return transformer_config


def convert_tokenizer_config(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tokenizer config from YAML to new format."""
    model_config = yaml_config.get("model", {})

    tokenizer_config = {
        "vocab_size": model_config.get("vocab_size", 100278),
        "eos_token_id": model_config.get("eos_token_id", 100257),
        "pad_token_id": model_config.get("pad_token_id", 100277),
        "identifier": "allenai/dolma2-tokenizer",
        "_CLASS_": "olmo_core.data.tokenizer.TokenizerConfig"
    }

    if "tokenizer" in yaml_config:
        tokenizer = yaml_config["tokenizer"]
        if "identifier" in tokenizer:
            tokenizer_config["identifier"] = tokenizer["identifier"]

    return tokenizer_config


def convert_yaml_to_json(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = convert_model_config(yaml_config["model"])
    tokenizer_config = convert_tokenizer_config(yaml_config)


    json_config = {
        "model": model_config,
        "dataset": {
            "tokenizer": tokenizer_config
        }
    }

    if "run_name" in yaml_config:
        json_config["run_name"] = yaml_config["run_name"]

    return json_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "yaml_config",
        help="Path to the input YAML config file"
    )
    parser.add_argument(
        "json_config",
        help="Path to the output JSON config file"
    )
    args = parser.parse_args()
    with open(args.yaml_config, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    json_config = convert_yaml_to_json(yaml_config)

    with open(args.json_config, 'w', encoding='utf-8') as f:
        json.dump(json_config, f, indent=2)

    print(f"Converted {args.yaml_config} to {args.json_config}")
    print(f"Model: {json_config['model']['d_model']}D, {json_config['model']['n_layers']} layers")
    print(f"Vocab size: {json_config['dataset']['tokenizer']['vocab_size']}")


if __name__ == "__main__":
    main()