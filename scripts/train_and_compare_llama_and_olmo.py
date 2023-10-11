import logging

import numpy as np
import torch

from olmo import TrainConfig, Olmo
from olmo.config import FSDPWrapStrategy
from olmo.model import OlmoGenerateOutput
from olmo.util import prepare_cli_environment
import transformers
import os
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed as dist
import torch.nn.functional as F


prepare_cli_environment()
log = logging.getLogger(__name__)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # needed for running in the deterministic mode

# for development
# hf_device = 'cpu'
# olmo_device = 'cpu'
# use_fsdp = False
# model_path = 'test_fixtures/tiny_llama/'

# # for running the real 7B model on GPU
# hf_device = 'cuda:0'
# olmo_device = 'cuda:1'
# use_fsdp = False
# model_path = '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B'

# # for FSDP
hf_device = "cpu"
olmo_device = "cuda"
use_fsdp = True
model_path = "test_fixtures/tiny_llama/"


def test_all_approx_close(a, b, rtol, atol, count):
    idx = torch.isclose(a, b, rtol, atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        log.error(
            f"Too many values ({sumval}/{a.numel()})=({100 * sumval // a.numel()}%) not close: test {sumval} < {count}"
        )


def get_world_size():
    return int(os.environ.get("WORLD_SIZE") or 1)


# tokenizer is in the same directory as the HF model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)


# load Llama weights into HF model
def build_hf_model(device=hf_device):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map=device, rms_norm_eps=1e-5)
    return hf_model


def non_meta_device(device_str):
    if device_str == "meta":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return torch.device(device_str)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK") or 0)


# enrich an olmo model with FSDP functionality
def apply_fsdp(olmo_model: Olmo, cfg: TrainConfig):
    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    wrap_policy = None
    if cfg.fsdp.wrapping_strategy == FSDPWrapStrategy.by_block:
        wrap_policy = olmo_model.fsdp_wrap_fn
    elif cfg.fsdp.wrapping_strategy == FSDPWrapStrategy.size_based:
        wrap_policy = size_based_auto_wrap_policy

    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=non_meta_device(cfg.model.init_device))

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None

    cfg.fsdp.use_orig_params = True

    torch.manual_seed(42)
    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=cfg.fsdp_precision,
        auto_wrap_policy=wrap_policy,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        param_init_fn=param_init_fn,
    )
    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if param_init_fn is not None:
        olmo_model.reset_parameters()

    return fsdp_model


# create a similar sized OLMo model
def build_olmo_model(hf_model, device=olmo_device, use_fsdp=False):
    cfg = TrainConfig.load("configs/v1_5-mix-medium-llama-local.yaml")
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    cfg.model.init_device = device

    cfg.model.n_layers = hf_model.config.num_hidden_layers
    cfg.model.n_heads = hf_model.config.num_attention_heads
    cfg.model.d_model = hf_model.config.hidden_size
    cfg.model.mlp_hidden_size = hf_model.config.intermediate_size * 2

    # Make model
    log.info("Building model...")
    olmo_model = Olmo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")

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
                [
                    hf_layer.self_attn.q_proj.weight,
                    hf_layer.self_attn.k_proj.weight,
                    hf_layer.self_attn.v_proj.weight,
                ],
                dim=0,
            )
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
            new_ff_proj = torch.cat([hf_layer.mlp.up_proj.weight, hf_layer.mlp.gate_proj.weight], dim=0)
            parameters_to_read.remove(f"model.layers.{i}.mlp.up_proj.weight")
            parameters_to_read.remove(f"model.layers.{i}.mlp.gate_proj.weight")
            assert new_ff_proj.shape == olmo_layer.ff_proj.weight.shape
            assert new_ff_proj.dtype == olmo_layer.ff_proj.weight.dtype
            olmo_layer.ff_proj.weight.copy_(new_ff_proj)
            parameters_to_set.remove(f"transformer.blocks.{i}.ff_proj.weight")

    # all done?
    assert len(parameters_to_set) == 0
    assert len(parameters_to_read) == 0

    if use_fsdp:
        olmo_model = apply_fsdp(olmo_model, cfg)

    return olmo_model


# Initialize process group and set device.
if use_fsdp:
    dist.init_process_group(backend="nccl", rank=get_local_rank(), world_size=get_world_size())
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")

hf_model = build_hf_model(device=hf_device)
olmo_model = build_olmo_model(hf_model, device=olmo_device, use_fsdp=use_fsdp)

# ## ========== uncomment one of the following to test if both models are the same type ==========

# # Test if both are OLMo models
# hf_model = build_olmo_model(hf_model, device=hf_device)

# # Test if both are HF models
# olmo_model = build_hf_model(device=olmo_device)


# run one batch
with open("scripts/spiky_batch.npy", "rb") as f:
    buffer = f.read()
array = np.frombuffer(buffer, dtype=np.uint64)
batch = torch.tensor(array.astype(np.int_), dtype=torch.long)
batch = batch.reshape(2048, -1)
batch = batch % 32000  # Llama vocab size is 32k
train_batch = batch[2:4, :50]
test_batch = batch[:2, :50]  # don't run all 4M tokens
test_string = "The sky's color is"


def generate(model, tokenizer, input_str):
    log.info(f"Generating from: {input_str}")
    tokens = tokenizer.encode(input_str, return_tensors="pt").to(device=non_meta_device(model.device))
    generated_ids = model.generate(tokens)

    if isinstance(generated_ids, OlmoGenerateOutput):
        generated_ids = generated_ids.token_ids

    token_ids = torch.flatten(generated_ids)
    log.info(f"Generated token ids: {token_ids}")
    return tokenizer.decode(token_ids)


# run on olmo
torch.manual_seed(42)
olmo_output = olmo_model(test_batch.to(device=non_meta_device(olmo_device)))
olmo_logits = olmo_output.logits
log.info(f"OLmo logits: {olmo_logits}")

# run on hf
torch.manual_seed(42)
hf_output = hf_model(test_batch.to(device=hf_device))
hf_logits = hf_output.logits
log.info(f"HF logits: {hf_logits}")
torch.manual_seed(42)
test_all_approx_close(olmo_logits.cpu().float(), hf_logits.cpu().float(), atol=1e-4, rtol=1e-3, count=10)
if not torch.allclose(olmo_logits.cpu().float(), hf_logits.cpu().float(), atol=1e-4, rtol=1e-3):
    log.error("Olmo and HF logits fail torch.allclose()")


def reformat_labels_to_look_like_logits(labels):
    # the labels are of size [batch_size, sequence_length] and contain values in [0, ..., config.vocab_size]
    # turn them into 1 hot vectors of size [batch_size, sequence_length, config.vocab_size] for cross entropy
    batch_size, sequence_length = labels.shape
    vocab_size = 32000
    one_hot_labels = torch.zeros((batch_size, sequence_length, vocab_size), dtype=torch.float32)
    for i in range(batch_size):
        for j in range(sequence_length):
            one_hot_labels[i, j, labels[i, j]] = 1
    return one_hot_labels


olmo_generation_model = olmo_model
if use_fsdp:
    log.warning(
        "Generate bypasses FSDP's forward implementation, which causes generation to fail. Using a CPU model instead."
    )
    olmo_generation_model = build_olmo_model(hf_model, device="cpu", use_fsdp=False)

log.info(f"OLMo generation: {generate(olmo_generation_model, tokenizer, test_string)}")
log.info(f"HF generation: {generate(hf_model, tokenizer, test_string)}")


def print_metrics(olmo_tensor, hf_tensor, tensor_name):
    log.info(f"OLMo {tensor_name} norm: {torch.norm(olmo_tensor)}")
    log.info(f"HF {tensor_name} norm: {torch.norm(hf_tensor)}")
    log.info(f"OLMo {tensor_name} mean: {torch.mean(olmo_tensor)}")
    log.info(f"HF {tensor_name} mean: {torch.mean(hf_tensor)}")
    log.info(f"OLMo {tensor_name} min: {torch.min(olmo_tensor)}")
    log.info(f"HF {tensor_name} min: {torch.min(hf_tensor)}")
    log.info(f"OLMo {tensor_name} max: {torch.max(olmo_tensor)}")
    log.info(f"HF {tensor_name} max: {torch.max(hf_tensor)}")


# train on batch
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
log.info("Training...")
olmo_optimzer = torch.optim.AdamW(olmo_model.parameters(), lr=1, betas=(0.9, 0.95))
hf_optimizer = torch.optim.AdamW(hf_model.parameters(), lr=1, betas=(0.9, 0.95))
for i in range(10):
    idx = 2 * (i + 1)
    train_batch = batch[idx : idx + 2, :50]
    labels = batch[idx + 1 : idx + 3, :50]
    labels = reformat_labels_to_look_like_logits(labels)

    olmo_optimzer.zero_grad()
    hf_optimizer.zero_grad()

    torch.manual_seed(42)
    olmo_logits = olmo_model(train_batch.to(device=olmo_device)).logits
    torch.manual_seed(42)
    hf_logits = hf_model(train_batch.to(device=hf_device)).logits
    torch.manual_seed(42)

    olmo_loss = F.cross_entropy(olmo_logits, labels.to(device=olmo_device))
    hf_loss = F.cross_entropy(hf_logits, labels.to(device=hf_device))

    torch.manual_seed(42)
    olmo_loss.backward()
    torch.manual_seed(42)
    hf_loss.backward()
    torch.manual_seed(42)

    olmo_optimzer.step()
    hf_optimizer.step()

    # run on olmo
    olmo_logits = olmo_model(test_batch.to(device=olmo_device)).logits
    log.info(f"OLMo logits: {olmo_logits}")
    hf_logits = hf_model(test_batch.to(device=hf_device)).logits
    log.info(f"HF logits: {hf_logits}")

    print_metrics(olmo_logits, hf_logits, "logits")
    try:
        olmo_input_embeddings = olmo_model.transformer.wte(test_batch.to(device=olmo_device))
        hf_input_embeddings = hf_model.model.embed_tokens(test_batch.to(device=hf_device))
        print_metrics(olmo_input_embeddings, hf_input_embeddings, "input embeddings")
        olmo_input_embedding_gradients = olmo_model.transformer.wte.weight.grad
        hf_input_embedding_gradients = hf_model.model.embed_tokens.weight.grad
        print_metrics(olmo_input_embedding_gradients, hf_input_embedding_gradients, "input embedding gradients")
    except Exception:
        pass
