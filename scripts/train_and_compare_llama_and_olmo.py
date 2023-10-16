import logging

import numpy as np
import torch

from olmo import TrainConfig, Olmo
from olmo.config import FSDPWrapStrategy
from olmo.model import OlmoGenerateOutput, OlmoOutput
from olmo.optim import build_optimizer
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

# torch.set_printoptions(precision=10)
SEED: int = 42
SEQ_LEN: int = 3

model_path = "test_fixtures/tiny_llama/"
# model_path = '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B'

# for development
# hf_device = 'cpu'
# olmo_device = 'cpu'
# use_fsdp = False

# # for running the real 7B model on GPU
# hf_device = 'cuda:0'
# olmo_device = 'cuda:1'
# use_fsdp = False

# # for FSDP
hf_device = "cpu"
olmo_device = "cuda"
use_fsdp = True


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
def build_hf_model(device: str, dtype: torch.dtype):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device, rms_norm_eps=1e-5
    )
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

    torch.manual_seed(SEED)
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


def build_config(device):
    cfg = TrainConfig.load("configs/v1_5-mix-medium-llama-local.yaml")
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    cfg.model.init_device = device

    return cfg


def update_config_with_hf_settings(cfg, hf_model):
    cfg.model.n_layers = hf_model.config.num_hidden_layers
    cfg.model.n_heads = hf_model.config.num_attention_heads
    cfg.model.d_model = hf_model.config.hidden_size
    cfg.model.mlp_hidden_size = hf_model.config.intermediate_size * 2
    cfg.model.max_sequence_length = hf_model.config.max_position_embeddings


# create a similar sized OLMo model
def build_olmo_model(hf_model, cfg, use_fsdp=False):
    # Make model
    log.info("Building model...")
    olmo_model = Olmo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")

    parameters_to_set = {name for name, _ in olmo_model.named_parameters()}
    parameters_to_read = {name for name, _ in hf_model.named_parameters()}

    with torch.no_grad():
        # embeddings
        # assert olmo_model.transformer.wte.weight.dtype == hf_model.model.embed_tokens.weight.dtype
        assert olmo_model.transformer.wte.weight.shape == hf_model.model.embed_tokens.weight.shape
        olmo_model.transformer.wte.weight.copy_(hf_model.model.embed_tokens.weight)
        parameters_to_set.remove("transformer.wte.weight")
        parameters_to_read.remove("model.embed_tokens.weight")

        # output projection
        assert hf_model.lm_head.weight.shape == olmo_model.transformer.ff_out.weight.shape
        # assert hf_model.lm_head.weight.dtype == olmo_model.transformer.ff_out.weight.dtype
        olmo_model.transformer.ff_out.weight.copy_(hf_model.lm_head.weight)
        parameters_to_set.remove("transformer.ff_out.weight")
        parameters_to_read.remove("lm_head.weight")

        # final layer norm
        assert hf_model.model.norm.weight.shape == olmo_model.transformer.ln_f.weight.shape
        # assert hf_model.model.norm.weight.dtype == olmo_model.transformer.ln_f.weight.dtype
        olmo_model.transformer.ln_f.weight.copy_(hf_model.model.norm.weight)
        parameters_to_set.remove("transformer.ln_f.weight")
        parameters_to_read.remove("model.norm.weight")

        # layers
        assert len(hf_model.model.layers) == len(olmo_model.transformer.blocks)
        for i, (hf_layer, olmo_layer) in enumerate(zip(hf_model.model.layers, olmo_model.transformer.blocks)):
            # input norm
            assert hf_layer.input_layernorm.weight.shape == olmo_layer.attn_norm.weight.shape
            # assert hf_layer.input_layernorm.weight.dtype == olmo_layer.attn_norm.weight.dtype
            olmo_layer.attn_norm.weight.copy_(hf_layer.input_layernorm.weight)
            parameters_to_set.remove(f"transformer.blocks.{i}.attn_norm.weight")
            parameters_to_read.remove(f"model.layers.{i}.input_layernorm.weight")

            # post attention layernorm
            assert hf_layer.post_attention_layernorm.weight.shape == olmo_layer.ff_norm.weight.shape
            # assert hf_layer.post_attention_layernorm.weight.dtype == olmo_layer.ff_norm.weight.dtype
            olmo_layer.ff_norm.weight.copy_(hf_layer.post_attention_layernorm.weight)
            parameters_to_set.remove(f"transformer.blocks.{i}.ff_norm.weight")
            parameters_to_read.remove(f"model.layers.{i}.post_attention_layernorm.weight")

            # q, k, v projections
            # TODO: We already know this does not produce the same result. It's close, but not close enough for
            # torch.allclose().
            # assert hf_layer.self_attn.q_proj.weight.dtype == olmo_layer.att_proj.weight.dtype
            # assert hf_layer.self_attn.k_proj.weight.dtype == olmo_layer.att_proj.weight.dtype
            # assert hf_layer.self_attn.v_proj.weight.dtype == olmo_layer.att_proj.weight.dtype
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
            # assert new_att_proj.dtype == olmo_layer.att_proj.weight.dtype
            olmo_layer.att_proj.weight.copy_(new_att_proj)
            parameters_to_set.remove(f"transformer.blocks.{i}.att_proj.weight")

            # attention out
            assert hf_layer.self_attn.o_proj.weight.shape == olmo_layer.attn_out.weight.shape
            # assert hf_layer.self_attn.o_proj.weight.dtype == olmo_layer.attn_out.weight.dtype
            olmo_layer.attn_out.weight.copy_(hf_layer.self_attn.o_proj.weight)
            parameters_to_set.remove(f"transformer.blocks.{i}.attn_out.weight")
            parameters_to_read.remove(f"model.layers.{i}.self_attn.o_proj.weight")

            # swiglu output projection
            assert hf_layer.mlp.down_proj.weight.shape == olmo_layer.ff_out.weight.shape
            # assert hf_layer.mlp.down_proj.weight.dtype == olmo_layer.ff_out.weight.dtype
            olmo_layer.ff_out.weight.copy_(hf_layer.mlp.down_proj.weight)
            parameters_to_set.remove(f"transformer.blocks.{i}.ff_out.weight")
            parameters_to_read.remove(f"model.layers.{i}.mlp.down_proj.weight")

            # swiglu input projections
            # TODO: If fused q, k, v above doesn't produce the same result, then this probably also doesn't.
            # assert hf_layer.mlp.up_proj.weight.dtype == olmo_layer.ff_proj.weight.dtype
            # assert hf_layer.mlp.gate_proj.weight.dtype == olmo_layer.ff_proj.weight.dtype
            new_ff_proj = torch.cat([hf_layer.mlp.up_proj.weight, hf_layer.mlp.gate_proj.weight], dim=0)
            parameters_to_read.remove(f"model.layers.{i}.mlp.up_proj.weight")
            parameters_to_read.remove(f"model.layers.{i}.mlp.gate_proj.weight")
            assert new_ff_proj.shape == olmo_layer.ff_proj.weight.shape
            # assert new_ff_proj.dtype == olmo_layer.ff_proj.weight.dtype
            olmo_layer.ff_proj.weight.copy_(new_ff_proj)
            parameters_to_set.remove(f"transformer.blocks.{i}.ff_proj.weight")

    # all done?
    assert len(parameters_to_set) == 0, parameters_to_set
    assert len(parameters_to_read) == 0, parameters_to_read

    if use_fsdp:
        olmo_model = apply_fsdp(olmo_model, cfg)

    return olmo_model


def print_metrics(olmo_tensor, hf_tensor, tensor_name):
    log.info(f"{tensor_name} max absolute diff: {torch.max(torch.abs(olmo_tensor.cpu() - hf_tensor.cpu()))}")
    log.info(
        f"{tensor_name} max relative diff: {torch.max(torch.abs(olmo_tensor.cpu() - hf_tensor.cpu()) / torch.abs(olmo_tensor.cpu()))}"
    )
    log.info(f"OLMo {tensor_name} norm: {torch.norm(olmo_tensor)}")
    log.info(f"HF {tensor_name} norm: {torch.norm(hf_tensor)}")
    log.info(f"OLMo {tensor_name} mean: {torch.mean(olmo_tensor)}")
    log.info(f"HF {tensor_name} mean: {torch.mean(hf_tensor)}")
    log.info(f"OLMo {tensor_name} min: {torch.min(olmo_tensor)}")
    log.info(f"HF {tensor_name} min: {torch.min(hf_tensor)}")
    log.info(f"OLMo {tensor_name} max: {torch.max(olmo_tensor)}")
    log.info(f"HF {tensor_name} max: {torch.max(hf_tensor)}")


def check_weight_equality(olmo_weight: torch.Tensor, hf_weight: torch.Tensor, tensor_name):
    if not torch.allclose(olmo_weight.cpu().float(), hf_weight.cpu().float()):
        log.warning("Weights not equivalent for %s", tensor_name)
        print_metrics(olmo_weight, hf_weight, tensor_name)
        return False

    return True


def check_model_equality(hf_model, olmo_model):
    are_equal = True

    # embeddings
    are_equal = (
        check_weight_equality(olmo_model.transformer.wte.weight, hf_model.model.embed_tokens.weight, "wte")
        and are_equal
    )

    # output projection
    are_equal = (
        check_weight_equality(olmo_model.transformer.ff_out.weight, hf_model.lm_head.weight, "ff_out")
        and are_equal
    )

    # final layer norm
    are_equal = (
        check_weight_equality(olmo_model.transformer.ln_f.weight, hf_model.model.norm.weight, "ln_f") and are_equal
    )

    # layers
    assert len(hf_model.model.layers) == len(olmo_model.transformer.blocks)
    for i, (hf_layer, olmo_layer) in enumerate(zip(hf_model.model.layers, olmo_model.transformer.blocks)):
        # input norm
        are_equal = (
            check_weight_equality(olmo_layer.attn_norm.weight, hf_layer.input_layernorm.weight, f"attn_norm_{i}")
            and are_equal
        )

        # post attention layernorm
        are_equal = (
            check_weight_equality(
                olmo_layer.ff_norm.weight, hf_layer.post_attention_layernorm.weight, f"attn_norm_{i}"
            )
            and are_equal
        )

        # q, k, v projections
        # TODO: We already know this does not produce the same result. It's close, but not close enough for
        # torch.allclose().
        q_proj, k_proj, v_proj = olmo_layer.att_proj.weight.chunk(3, dim=0)
        are_equal = check_weight_equality(q_proj, hf_layer.self_attn.q_proj.weight, f"q_proj_{i}") and are_equal
        are_equal = check_weight_equality(k_proj, hf_layer.self_attn.k_proj.weight, f"k_proj_{i}") and are_equal
        are_equal = check_weight_equality(v_proj, hf_layer.self_attn.v_proj.weight, f"v_proj_{i}") and are_equal
        # print_metrics(q_proj, hf_layer.self_attn.q_proj.weight, f"q_proj_{i}")
        # print_metrics(k_proj, hf_layer.self_attn.k_proj.weight, f"k_proj_{i}")
        # print_metrics(v_proj, hf_layer.self_attn.v_proj.weight, f"v_proj_{i}")

        # attention out
        are_equal = (
            check_weight_equality(olmo_layer.attn_out.weight, hf_layer.self_attn.o_proj.weight, f"attn_out_{i}")
            and are_equal
        )

        # swiglu output projection
        are_equal = (
            check_weight_equality(olmo_layer.ff_out.weight, hf_layer.mlp.down_proj.weight, f"ff_out_{i}")
            and are_equal
        )

        # swiglu input projections
        # TODO: If fused q, k, v above doesn't produce the same result, then this probably also doesn't.
        up_proj, gate_proj = olmo_layer.ff_proj.weight.chunk(2, dim=0)
        are_equal = check_weight_equality(up_proj, hf_layer.mlp.up_proj.weight, f"up_proj_{i}") and are_equal
        are_equal = check_weight_equality(gate_proj, hf_layer.mlp.gate_proj.weight, f"gate_proj_{i}") and are_equal
        # print_metrics(up_proj, hf_layer.mlp.up_proj.weight, f"up_proj_{i}")
        # print_metrics(gate_proj, hf_layer.mlp.gate_proj.weight, f"gate_proj_{i}")

    assert are_equal, "Model weight equality check failed"


def check_grad_equality(hf_model, olmo_model):
    are_equal = True
    # Check in reverse order

    # output projection
    are_equal = (
        check_weight_equality(
            olmo_model.transformer.ff_out.weight.grad, hf_model.lm_head.weight.grad, "ff_out_grad"
        )
        and are_equal
    )

    # final layer norm
    are_equal = (
        check_weight_equality(
            olmo_model.transformer.ln_f.weight.grad, hf_model.model.norm.weight.grad, "ln_f_grad"
        )
        and are_equal
    )

    # layers
    assert len(hf_model.model.layers) == len(olmo_model.transformer.blocks)
    for i, (hf_layer, olmo_layer) in reversed(
        list(enumerate(zip(hf_model.model.layers, olmo_model.transformer.blocks)))
    ):
        # swiglu output projection
        are_equal = (
            check_weight_equality(olmo_layer.ff_out.weight.grad, hf_layer.mlp.down_proj.weight.grad, "ff_out_grad")
            and are_equal
        )

        # swiglu input projections
        # TODO: If fused q, k, v above doesn't produce the same result, then this probably also doesn't.
        up_proj_grad, gate_proj_grad = olmo_layer.ff_proj.weight.grad.chunk(2, dim=0)
        are_equal = (
            check_weight_equality(up_proj_grad, hf_layer.mlp.up_proj.weight.grad, f"up_proj_grad_{i}")
            and are_equal
        )
        are_equal = (
            check_weight_equality(gate_proj_grad, hf_layer.mlp.gate_proj.weight.grad, f"gate_proj_grad_{i}")
            and are_equal
        )
        # print_metrics(up_proj_grad, hf_layer.mlp.up_proj.weight.grad, f"up_proj_grad_{i}")
        # print_metrics(gate_proj_grad, hf_layer.mlp.gate_proj.weight.grad, f"gate_proj_grad_{i}")

        # attention out
        are_equal = (
            check_weight_equality(
                olmo_layer.attn_out.weight.grad, hf_layer.self_attn.o_proj.weight.grad, f"attn_out_grad_{i}"
            )
            and are_equal
        )

        # post attention layernorm
        are_equal = (
            check_weight_equality(
                olmo_layer.ff_norm.weight.grad, hf_layer.post_attention_layernorm.weight.grad, f"ff_norm_grad_{i}"
            )
            and are_equal
        )

        # q, k, v projections
        # TODO: We already know this does not produce the same result. It's close, but not close enough for
        # torch.allclose().
        q_proj_grad, k_proj_grad, v_proj_grad = olmo_layer.att_proj.weight.grad.chunk(3, dim=0)
        are_equal = (
            check_weight_equality(q_proj_grad, hf_layer.self_attn.q_proj.weight.grad, f"q_proj_grad_{i}")
            and are_equal
        )
        are_equal = (
            check_weight_equality(k_proj_grad, hf_layer.self_attn.k_proj.weight.grad, f"k_proj_grad_{i}")
            and are_equal
        )
        are_equal = (
            check_weight_equality(v_proj_grad, hf_layer.self_attn.v_proj.weight.grad, f"v_proj_grad_{i}")
            and are_equal
        )
        # print_metrics(q_proj_grad, hf_layer.self_attn.q_proj.weight.grad, f"q_proj_grad_{i}")
        # print_metrics(k_proj_grad, hf_layer.self_attn.k_proj.weight.grad, f"k_proj_grad_{i}")
        # print_metrics(v_proj_grad, hf_layer.self_attn.v_proj.weight.grad, f"v_proj_grad_{i}")

        # input norm
        are_equal = (
            check_weight_equality(
                olmo_layer.attn_norm.weight.grad, hf_layer.input_layernorm.weight.grad, f"attn_norm_grad_{i}"
            )
            and are_equal
        )

    # embeddings
    are_equal = (
        check_weight_equality(
            olmo_model.transformer.wte.weight.grad, hf_model.model.embed_tokens.weight.grad, "wte_grad"
        )
        and are_equal
    )

    assert are_equal, "Grad equality check failed"


config = build_config(olmo_device)

# Initialize process group and set device.
if use_fsdp:
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")

hf_model = build_hf_model(hf_device, config.autocast_precision)
update_config_with_hf_settings(config, hf_model)
olmo_model = build_olmo_model(hf_model, config, use_fsdp=use_fsdp)

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
train_batch = batch[2:4, :SEQ_LEN]
test_batch = batch[:2, :SEQ_LEN]  # don't run all 4M tokens
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


def olmo_forward(olmo_model: torch.nn.Module, batch: torch.Tensor, autocast_dtype: torch.dtype) -> OlmoOutput:
    device_type = non_meta_device(olmo_device)
    if autocast_dtype != torch.float32:
        with torch.autocast(device_type.type, dtype=autocast_dtype):
            return olmo_model(batch.to(device_type))

    else:
        return olmo_model(batch.to(device_type))


# run on olmo
torch.manual_seed(SEED)
olmo_output = olmo_forward(olmo_model, test_batch, config.autocast_precision)
olmo_logits = olmo_output.logits
log.info(f"OLmo logits: {olmo_logits}")

# run on hf
torch.manual_seed(SEED)
hf_output = hf_model(test_batch.to(device=hf_device))
hf_logits = hf_output.logits
log.info(f"HF logits: {hf_logits}")

print_metrics(olmo_logits, hf_logits, "logits")
if not use_fsdp:
    check_model_equality(hf_model, olmo_model)

torch.manual_seed(SEED)
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
    config.model.init_device = "cpu"
    olmo_generation_model = build_olmo_model(hf_model, config, use_fsdp=False)
    config.model.init_device = olmo_device

log.info(f"OLMo generation: {generate(olmo_generation_model, tokenizer, test_string)}")
log.info(f"HF generation: {generate(hf_model, tokenizer, test_string)}")

# train on batch
torch.use_deterministic_algorithms(True)
torch.manual_seed(SEED)
log.info("Training...")

# config.optimizer.learning_rate = 0.1
olmo_optimizer = build_optimizer(config, olmo_model)
hf_optimizer = build_optimizer(config, hf_model)

# betas = (0.9, 0.95)
# # betas = (0., 0.)
# eps = 1e-5
# lr = 3.0e-4
# weight_decay = 0.1
# olmo_optimizer = torch.optim.AdamW(olmo_model.parameters(), weight_decay=weight_decay, lr=lr, betas=betas, eps=eps)
# hf_optimizer = torch.optim.AdamW(hf_model.parameters(), weight_decay=weight_decay, lr=lr, betas=betas, eps=eps)

# olmo_optimizer = torch.optim.Adamax(olmo_model.parameters(), lr=1.)
# hf_optimizer = torch.optim.Adamax(hf_model.parameters(), lr=1.)

# olmo_optimizer = torch.optim.SGD(olmo_model.parameters(), lr=0.1)
# hf_optimizer = torch.optim.SGD(hf_model.parameters(), lr=0.1)
for i in range(10):
    log.info("Training iteration %d", i + 1)

    idx = 2 * (i + 1)
    train_batch = batch[idx : idx + 2, :SEQ_LEN]
    labels = batch[idx + 1 : idx + 3, :SEQ_LEN]
    labels = reformat_labels_to_look_like_logits(labels)

    olmo_optimizer.zero_grad()
    hf_optimizer.zero_grad()

    torch.manual_seed(SEED)
    olmo_logits = olmo_forward(olmo_model, train_batch, config.autocast_precision).logits
    # olmo_logits = olmo_model(train_batch.to(device=olmo_device)).logits
    torch.manual_seed(SEED)
    hf_logits = hf_model(train_batch.to(device=hf_device)).logits

    if not use_fsdp:
        check_model_equality(hf_model, olmo_model)
    print_metrics(olmo_logits, hf_logits, "logits")

    torch.manual_seed(SEED)
    olmo_loss = F.cross_entropy(olmo_logits, labels.to(device=olmo_device))
    torch.manual_seed(SEED)
    hf_loss = F.cross_entropy(hf_logits, labels.to(device=hf_device))

    torch.manual_seed(SEED)
    olmo_loss.backward()
    torch.manual_seed(SEED)
    hf_loss.backward()
    torch.manual_seed(SEED)

    if not use_fsdp:
        check_grad_equality(hf_model, olmo_model)

    log.info(f"OLMo loss: {olmo_loss}")
    log.info(f"HF loss: {hf_loss}")

    torch.manual_seed(SEED)
    olmo_optimizer.step()
    torch.manual_seed(SEED)
    hf_optimizer.step()

    if not use_fsdp:
        check_model_equality(hf_model, olmo_model)

    # run on olmo
    olmo_logits = olmo_forward(olmo_model, test_batch, config.autocast_precision).logits
    # olmo_logits = olmo_model(test_batch.to(device=olmo_device)).logits
    log.info(f"OLMo logits: {olmo_logits}")
    hf_logits = hf_model(test_batch.to(device=hf_device)).logits
    log.info(f"HF logits: {hf_logits}")

    # print_metrics(olmo_logits, hf_logits, "logits")
    # try:
    #     olmo_input_embeddings = olmo_model.transformer.wte(test_batch.to(device=olmo_device))
    #     hf_input_embeddings = hf_model.model.embed_tokens(test_batch.to(device=hf_device))
    #     print_metrics(olmo_input_embeddings, hf_input_embeddings, "input embeddings")
    #     olmo_input_embedding_gradients = olmo_model.transformer.wte.weight.grad
    #     hf_input_embedding_gradients = hf_model.model.embed_tokens.weight.grad
    #     print_metrics(olmo_input_embedding_gradients, hf_input_embedding_gradients, "input embedding gradients")
    # except Exception:
    #     pass
