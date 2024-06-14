import math
from typing import List, Optional

import pytest
import torch.nn
from torch.testing import assert_close

from olmo.config import InitFnType, ModelConfig
from olmo.model import BufferCache, OLMo, OLMoBlock, OLMoLlamaBlock, OLMoSequentialBlock
from olmo.torch_util import seed_all


def check_distribution(
    module: torch.nn.Module,
    mean: float,
    std: float,
    max_val: Optional[float] = None,
    min_val: Optional[float] = None,
    bias_should_be_zero: bool = True,
    diff: float = 1e-4,
    ignore_params: Optional[List] = None,
):
    for name, param in module.named_parameters():
        if ignore_params is not None and any([ignored in name for ignored in ignore_params]):
            print(f"ignoring {name}")
            continue
        if "bias" in name and bias_should_be_zero:
            expected_mean = 0.0
            expected_std = 0.0
        else:
            expected_mean = mean
            expected_std = std

        actual_mean = param.data.mean()
        actual_std = param.data.std()

        assert_close(
            actual_mean,
            torch.tensor(expected_mean),
            atol=diff,
            rtol=diff,
            msg=f"Expected mean value for {name} = {expected_mean}, actual = {actual_mean}",
        )
        assert_close(
            actual_std,
            torch.tensor(expected_std),
            atol=diff,
            rtol=diff,
            msg=f"Expected std value for {name} = {expected_std}, actual = {actual_std}",
        )

        if max_val is not None:
            assert param.data.max() <= max_val
        if min_val is not None:
            assert param.data.min() >= min_val


#################################################################################
################################### OLMoBlock ###################################
#################################################################################


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_block_init_normal(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################
    cache = BufferCache()
    base_config = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.normal, init_std=0.02
    )

    for layer_id in [0, 4]:
        block = OLMoBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()
        check_distribution(block, 0.00, 0.02)

    ########################################### Truncated Normal init ###########################################
    base_config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        init_fn=InitFnType.normal,
        init_std=0.02,
        init_cutoff_factor=3.0,
    )
    block = OLMoBlock(layer_id=0, config=base_config, cache=cache)
    block.reset_parameters()

    check_distribution(block, 0.00, 0.02, 3.0 * 0.02, -3.0 * 0.02, diff=1e-3)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_block_init_mitchell(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2
    ################################################ Mitchell init ################################################

    cache = BufferCache()
    base_config = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.mitchell)

    # expected_std = 1/(math.sqrt(2*d*(layer_id+1))

    for layer_id in [0, 4]:
        block = OLMoBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()

        check_distribution(block.attn_out, 0.00, 1 / (math.sqrt(2 * d_model * (layer_id + 1))), diff=1e-3)
        check_distribution(
            block.ff_out, 0.00, 1 / (math.sqrt(2 * block.ff_out.in_features * (layer_id + 1))), diff=1e-3
        )


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_block_init_full_megatron(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Megatron init ################################################
    cache = BufferCache()

    for init_cutoff_factor in [None, 3]:
        base_config = ModelConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            init_fn=InitFnType.full_megatron,
            init_std=0.006,
            init_cutoff_factor=init_cutoff_factor,
        )

        for layer_id in [0, 4]:
            block = OLMoBlock(layer_id=layer_id, config=base_config, cache=cache)
            block.reset_parameters()

            check_distribution(
                block.attn_out, 0.00, 0.006 / math.sqrt(2.0 * n_layers), 3.0 * 0.006, -3.0 * 0.006, diff=1e-3
            )
            check_distribution(
                block.ff_out, 0.00, 0.006 / math.sqrt(2.0 * n_layers), 3.0 * 0.006, -3.0 * 0.006, diff=1e-3
            )


#################################################################################
############################## OLMoSequentialBlock ##############################
#################################################################################


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_sequential_block_init_normal(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################
    cache = BufferCache()
    base_config = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.normal, init_std=0.02
    )

    for layer_id in [0, 4]:
        block = OLMoSequentialBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()

        check_distribution(block, 0.00, 0.02, ignore_params=["attn_norm", "ff_norm"])
        # if parametric layer norm
        check_distribution(block.attn_norm, 1.00, 0.00)
        check_distribution(block.ff_norm, 1.00, 0.00)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_sequential_block_init_mitchell(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Mitchell init ################################################
    cache = BufferCache()
    base_config = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.mitchell)

    for layer_id in [0, 4]:
        block = OLMoSequentialBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()

        check_distribution(block.att_proj, 0.0, 1 / math.sqrt(d_model), diff=1e-3)
        check_distribution(block.ff_proj, 0.0, 1 / math.sqrt(d_model), diff=1e-3)

        check_distribution(block.attn_out, 0.00, 1 / (math.sqrt(2 * d_model * (layer_id + 1))), diff=1e-3)
        check_distribution(
            block.ff_out, 0.00, 1 / (math.sqrt(2 * block.ff_out.in_features * (layer_id + 1))), diff=1e-3
        )
        # if parametric layer norm
        check_distribution(block.attn_norm, 1.00, 0.00)
        check_distribution(block.ff_norm, 1.00, 0.00)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_sequential_block_init_full_megatron(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Megatron init ################################################
    cache = BufferCache()
    base_config = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.full_megatron, init_std=0.006
    )

    for layer_id in [0, 4]:
        block = OLMoSequentialBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()

        check_distribution(block.attn_out, 0.00, 0.006 / math.sqrt(2.0 * n_layers))
        check_distribution(block.ff_out, 0.00, 0.006 / math.sqrt(2.0 * n_layers))

        check_distribution(block.att_proj, 0.00, 0.006)
        check_distribution(block.ff_proj, 0.00, 0.006)

        # if parametric layer norm
        check_distribution(block.attn_norm, 1.00, 0.00)
        check_distribution(block.ff_norm, 1.00, 0.00)


#################################################################################
################################ OLMoLlamaBlock #################################
#################################################################################


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_llama_block_init_normal(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################
    cache = BufferCache()
    base_config = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.normal, init_std=0.02
    )

    for layer_id in [0, 4]:
        block = OLMoLlamaBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()

        check_distribution(block, 0.00, 0.02, ignore_params=["attn_norm", "ff_norm"])
        # if parametric layer norm
        check_distribution(block.attn_norm, 1.00, 0.00)
        check_distribution(block.ff_norm, 1.00, 0.00)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_llama_block_init_mitchell(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2
    ################################################ Mitchell init ################################################

    cache = BufferCache()
    base_config = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.mitchell)

    for layer_id in [0, 4]:
        block = OLMoLlamaBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()

        check_distribution(
            block,
            0.00,
            1 / math.sqrt(d_model),
            ignore_params=["attn_out", "ff_out", "attn_norm", "ff_norm"],
            diff=1e-3,
        )
        # if parametric layer norm
        check_distribution(block.attn_norm, 1.00, 0.00)
        check_distribution(block.ff_norm, 1.00, 0.00)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_llama_block_init_full_megatron(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Megatron init ################################################
    cache = BufferCache()
    base_config = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn=InitFnType.full_megatron, init_std=0.006
    )

    for layer_id in [0, 4]:
        block = OLMoLlamaBlock(layer_id=layer_id, config=base_config, cache=cache)
        block.reset_parameters()

        check_distribution(block.attn_out, 0.00, 0.006 / math.sqrt(2.0 * n_layers))
        check_distribution(block.ff_out, 0.00, 0.006 / math.sqrt(2.0 * n_layers))

        check_distribution(block.q_proj, 0.00, 0.006)
        check_distribution(block.k_proj, 0.00, 0.006)
        check_distribution(block.v_proj, 0.00, 0.006)
        check_distribution(block.ff_proj, 0.00, 0.006)

        # if parametric layer norm
        check_distribution(block.attn_norm, 1.00, 0.00)
        check_distribution(block.ff_norm, 1.00, 0.00)


#################################################################################
##################################### OLMo ######################################
#################################################################################


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_init_normal(seed: int):
    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################

    base_config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        init_fn=InitFnType.normal,
        init_std=0.02,
        weight_tying=False,
    )
    module = OLMo(config=base_config, init_params=True)

    check_distribution(module, 0.0, 0.02, ignore_params=["ln_f", "attn_norm", "ff_norm"])
    for i in range(n_layers):
        check_distribution(module.transformer.blocks[i].attn_norm, 1.00, 0.00)
        check_distribution(module.transformer.blocks[i].ff_norm, 1.00, 0.00)
    check_distribution(module.transformer.ln_f, 1.00, 0.00)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_init_mitchell(seed: int):
    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################

    base_config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        init_fn=InitFnType.mitchell,
        weight_tying=False,
    )
    module = OLMo(config=base_config, init_params=True)

    check_distribution(
        module.transformer,
        0.00,
        1 / math.sqrt(d_model),
        ignore_params=["attn_out", "ff_out", "attn_norm", "ff_norm", "ln_f"],
        diff=1e-3,
    )

    for i in range(n_layers):
        check_distribution(
            module.transformer.blocks[i].attn_out, 0.00, 1 / (math.sqrt(2 * d_model * (i + 1))), diff=1e-3
        )
        check_distribution(
            module.transformer.blocks[i].ff_out,
            0.00,
            1 / (math.sqrt(2 * module.transformer.blocks[i].ff_out.in_features * (i + 1))),
            diff=1e-3,
        )

        check_distribution(module.transformer.blocks[i].attn_norm, 1.00, 0.00)
        check_distribution(module.transformer.blocks[i].ff_norm, 1.00, 0.00)

    check_distribution(module.transformer.ln_f, 1.00, 0.00)
    check_distribution(module.transformer.ff_out, 0.00, 1 / math.sqrt(d_model), diff=1e-3)
    check_distribution(module.transformer.wte, 0.0, 1 / math.sqrt(d_model), diff=1e-3)
    check_distribution(module.transformer.wpe, 0.0, 1 / math.sqrt(d_model), diff=1e-3)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_init_full_megatron(seed: int):
    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Megatron init ################################################

    base_config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        init_fn=InitFnType.full_megatron,
        init_std=0.006,
        scale_logits=False,
        weight_tying=False,
    )
    module = OLMo(config=base_config, init_params=True)

    for i in range(n_layers):
        check_distribution(module.transformer.blocks[i].att_proj, 0.00, 0.006)
        check_distribution(module.transformer.blocks[i].ff_proj, 0.00, 0.006)
        check_distribution(module.transformer.blocks[i].attn_out, 0.00, 0.006 / math.sqrt(2 * n_layers))
        check_distribution(module.transformer.blocks[i].ff_out, 0.00, 0.006 / math.sqrt(2 * n_layers))
        check_distribution(module.transformer.blocks[i].attn_norm, 1.00, 0.00)
        check_distribution(module.transformer.blocks[i].ff_norm, 1.00, 0.00)
    check_distribution(module.transformer.ln_f, 1.00, 0.00)
    check_distribution(module.transformer.ff_out, 0.00, d_model**-0.5, diff=1e-3)

    check_distribution(module.transformer.wte, 0.0, 0.006, diff=1e-3)
    check_distribution(module.transformer.wpe, 0.0, 0.006)
