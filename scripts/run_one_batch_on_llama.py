import logging

import numpy as np
import torch

from olmo import TrainConfig, Olmo
from olmo.util import prepare_cli_environment

prepare_cli_environment()
log = logging.getLogger(__name__)


def get_world_size():
    return 1


# Load and fix config
cfg = TrainConfig.load("configs/v1_5-mix-medium-llama-local.yaml")
cfg.model.precision = cfg.precision
cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
assert cfg.device_train_batch_size is not None  # for mypy
cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
cfg.model.init_device = "cpu"

cfg.model.n_layers = 2      # for debugging

# Make model
log.info("Building model...")
olmo_model = Olmo(cfg.model)
log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")

# load Llama weights into Olmo
import transformers
hf_model = transformers.AutoModelForCausalLM.from_pretrained("/Users/dirkg/Documents/hf_llama2_models/7B")
del hf_model.model.layers[2:]  # Ananya's trick
parameters_to_set = {name for name, _ in olmo_model.named_parameters()}
parameters_to_read = {name for name, _ in hf_model.named_parameters()}

with torch.no_grad():
    # embeddings
    assert olmo_model.transformer.wte.weight.dtype == hf_model.model.embed_tokens.weight.dtype
    assert olmo_model.transformer.wte.weight.shape == hf_model.model.embed_tokens.weight.shape
    olmo_model.transformer.wte.weight.copy_(hf_model.model.embed_tokens.weight)
    parameters_to_set.remove("transformer.wte.weight")
    parameters_to_read.remove("model.embed_tokens.weight")

    # output projection
    assert hf_model.lm_head.weight.shape == olmo_model.transformer.ff_out.weight.shape
    assert hf_model.lm_head.weight.dtype == olmo_model.transformer.ff_out.weight.dtype
    olmo_model.transformer.ff_out.weight.copy_(hf_model.lm_head.weight)
    parameters_to_set.remove("transformer.ff_out.weight")
    parameters_to_read.remove("lm_head.weight")

    # final layer norm
    assert hf_model.model.norm.weight.shape == olmo_model.transformer.ln_f.weight.shape
    assert hf_model.model.norm.weight.dtype == olmo_model.transformer.ln_f.weight.dtype
    olmo_model.transformer.ln_f.weight.copy_(hf_model.model.norm.weight)
    parameters_to_set.remove("transformer.ln_f.weight")
    parameters_to_read.remove("model.norm.weight")

    # layers
    assert len(hf_model.model.layers) == len(olmo_model.transformer.blocks)
    for i, (hf_layer, olmo_layer) in enumerate(zip(hf_model.model.layers, olmo_model.transformer.blocks)):
        # input norm
        assert hf_layer.input_layernorm.weight.shape == olmo_layer.attn_norm.weight.shape
        assert hf_layer.input_layernorm.weight.dtype == olmo_layer.attn_norm.weight.dtype
        olmo_layer.attn_norm.weight.copy_(hf_layer.input_layernorm.weight)
        parameters_to_set.remove(f"transformer.blocks.{i}.attn_norm.weight")
        parameters_to_read.remove(f"model.layers.{i}.input_layernorm.weight")

        # post attention layernorm
        assert hf_layer.post_attention_layernorm.weight.shape == olmo_layer.ff_norm.weight.shape
        assert hf_layer.post_attention_layernorm.weight.dtype == olmo_layer.ff_norm.weight.dtype
        olmo_layer.ff_norm.weight.copy_(hf_layer.post_attention_layernorm.weight)
        parameters_to_set.remove(f"transformer.blocks.{i}.ff_norm.weight")
        parameters_to_read.remove(f"model.layers.{i}.post_attention_layernorm.weight")

        # q, k, v projections
        # TODO: We already know this does not produce the same result. It's close, but not close enough for
        # torch.allclose().
        assert hf_layer.self_attn.q_proj.weight.dtype == olmo_layer.att_proj.weight.dtype
        assert hf_layer.self_attn.k_proj.weight.dtype == olmo_layer.att_proj.weight.dtype
        assert hf_layer.self_attn.v_proj.weight.dtype == olmo_layer.att_proj.weight.dtype
        new_att_proj = torch.cat(
            [hf_layer.self_attn.q_proj.weight, hf_layer.self_attn.k_proj.weight, hf_layer.self_attn.v_proj.weight],
            dim=0)
        parameters_to_read.remove(f"model.layers.{i}.self_attn.q_proj.weight")
        parameters_to_read.remove(f"model.layers.{i}.self_attn.k_proj.weight")
        parameters_to_read.remove(f"model.layers.{i}.self_attn.v_proj.weight")
        assert new_att_proj.shape == olmo_layer.att_proj.weight.shape
        assert new_att_proj.dtype == olmo_layer.att_proj.weight.dtype
        olmo_layer.att_proj.weight.copy_(new_att_proj)
        parameters_to_set.remove(f"transformer.blocks.{i}.att_proj.weight")

        # attention out
        assert hf_layer.self_attn.o_proj.weight.shape == olmo_layer.attn_out.weight.shape
        assert hf_layer.self_attn.o_proj.weight.dtype == olmo_layer.attn_out.weight.dtype
        olmo_layer.attn_out.weight.copy_(hf_layer.self_attn.o_proj.weight)
        parameters_to_set.remove(f"transformer.blocks.{i}.attn_out.weight")
        parameters_to_read.remove(f"model.layers.{i}.self_attn.o_proj.weight")

        # swiglu output projection
        assert hf_layer.mlp.down_proj.weight.shape == olmo_layer.ff_out.weight.shape
        assert hf_layer.mlp.down_proj.weight.dtype == olmo_layer.ff_out.weight.dtype
        olmo_layer.ff_out.weight.copy_(hf_layer.mlp.down_proj.weight)
        parameters_to_set.remove(f"transformer.blocks.{i}.ff_out.weight")
        parameters_to_read.remove(f"model.layers.{i}.mlp.down_proj.weight")

        # swiglu input projections
        # TODO: If fused q, k, v above doesn't produce the same result, then this probably also doesn't.
        assert hf_layer.mlp.up_proj.weight.dtype == olmo_layer.ff_proj.weight.dtype
        assert hf_layer.mlp.gate_proj.weight.dtype == olmo_layer.ff_proj.weight.dtype
        new_ff_proj = torch.cat(
            [hf_layer.mlp.up_proj.weight, hf_layer.mlp.gate_proj.weight],
            dim=0)
        parameters_to_read.remove(f"model.layers.{i}.mlp.up_proj.weight")
        parameters_to_read.remove(f"model.layers.{i}.mlp.gate_proj.weight")
        assert new_ff_proj.shape == olmo_layer.ff_proj.weight.shape
        assert new_ff_proj.dtype == olmo_layer.ff_proj.weight.dtype
        olmo_layer.ff_proj.weight.copy_(new_ff_proj)
        parameters_to_set.remove(f"transformer.blocks.{i}.ff_proj.weight")

# all done?
assert len(parameters_to_set) == 0
assert len(parameters_to_read) == 0

# run one batch
with open("/Users/dirkg/Documents/transformers/scratch/spiky_batch.npy", "rb") as f:
    buffer = f.read()
array = np.frombuffer(buffer, dtype=np.uint64)
batch = torch.tensor(array.astype(np.int_), dtype=torch.long)
batch = batch.reshape(2048, -1)
batch = batch % 32000  # Llama vocab size is 32k
batch = batch[:2, :50]  # don't run all 4M tokens

output = olmo_model(batch)
logits = output.logits