import math
from typing import List, Optional

import pytest
import torch.nn
from torch.testing import assert_close

from olmo.config import ModelConfig
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
        if "bias" in name and bias_should_be_zero:
            expected_mean = 0.0
            expected_std = 0.0
        else:
            expected_mean = mean
            expected_std = std

        if ignore_params is not None and any([ignored in name for ignored in ignore_params]):
            print(f"ignoring {name}")
            continue

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


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_block_init(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################
    cache = BufferCache()
    base_config = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn="normal", init_std=0.02)

    block = OLMoBlock(layer_id=0, config=base_config, cache=cache)
    block.reset_parameters()

    check_distribution(block.attn_out, 0.00, 0.02)
    # TODO: confirm extra divisor; we may not want this.
    check_distribution(block.ff_out, 0.00, 0.02 / math.sqrt(2 * n_layers))

    # layer_id should make no difference for normal init
    block = OLMoBlock(layer_id=4, config=base_config, cache=cache)
    block.reset_parameters()
    check_distribution(block.attn_out, 0.00, 0.02)
    check_distribution(block.ff_out, 0.00, 0.02 / math.sqrt(2 * n_layers))

    ## truncated normal init
    base_config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        init_fn="normal",
        init_std=0.02,
        init_cutoff_factor=3.0,
    )
    block = OLMoBlock(layer_id=0, config=base_config, cache=cache)
    block.reset_parameters()

    # TODO: why is the diff higher?
    check_distribution(block.attn_out, 0.00, 0.02, 3.0*0.02, -3.0*0.02, diff=1e-3)
    check_distribution(block.ff_out, 0.00, 0.02 / math.sqrt(2 * n_layers), 3.0*0.02, -3.0*0.02, diff=1e-3)

    ## full_megatron init
    # base_config = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn="full_megatron", init_std=0.02)
    # block = OLMoBlock(layer_id=0, config=base_config, cache=cache)
    # block.reset_parameters()
    # check_distribution(block, 0.0, 0.02)

    ## mitchell init

    ## Scale embedding


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_sequential_block_init(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################
    cache = BufferCache()
    base_config = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn="normal", init_std=0.02)

    block = OLMoSequentialBlock(layer_id=0, config=base_config, cache=cache)
    block.reset_parameters()

    check_distribution(block.attn_out, 0.00, 0.02)
    check_distribution(block.ff_out, 0.00, 0.02 / math.sqrt(2 * n_layers))
    check_distribution(block.att_proj, 0.00, 0.02)
    check_distribution(block.ff_proj, 0.00, 0.02)
    # if parametric layer norm
    check_distribution(block.attn_norm, 1.00, 0.00)
    check_distribution(block.ff_norm, 1.00, 0.00)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_llama_block_init(seed: int):
    seed_all(seed)

    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################
    cache = BufferCache()
    base_config = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, init_fn="normal", init_std=0.02)

    block = OLMoLlamaBlock(layer_id=0, config=base_config, cache=cache)
    block.reset_parameters()

    check_distribution(block.attn_out, 0.00, 0.02)
    check_distribution(block.ff_out, 0.00, 0.02 / math.sqrt(2 * n_layers))
    check_distribution(block.q_proj, 0.00, 0.02)
    check_distribution(block.k_proj, 0.00, 0.02)
    check_distribution(block.v_proj, 0.00, 0.02)
    check_distribution(block.ff_proj, 0.00, 0.02)
    # if parametric layer norm
    check_distribution(block.attn_norm, 1.00, 0.00)
    check_distribution(block.ff_norm, 1.00, 0.00)


@pytest.mark.parametrize("seed", list(torch.randint(1, 10000, (3,))))
def test_olmo_init(seed: int):
    d_model = 1024
    n_heads = 2
    n_layers = 2

    ################################################ Normal init ################################################

    base_config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        init_fn="normal",
        init_std=0.02,
        scale_logits=False,
        weight_tying=False,
    )
    module = OLMo(config=base_config, init_params=True)

    check_distribution(module, 0.0, 0.02, ignore_params=["ff_out", "ln_f", "attn_norm", "ff_norm"])
    for i in range(n_layers):
        check_distribution(module.transformer.blocks[i].ff_out, 0.00, 0.02 / math.sqrt(2 * n_layers))
        check_distribution(module.transformer.blocks[i].attn_norm, 1.00, 0.00)
        check_distribution(module.transformer.blocks[i].ff_norm, 1.00, 0.00)
    check_distribution(module.transformer.ln_f, 1.00, 0.00)
    check_distribution(module.transformer.ff_out, 0.00, 0.02)

    # scale logits
    base_config = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        init_fn="normal",
        init_std=0.02,
        scale_logits=True,
        weight_tying=False,
    )
    module = OLMo(config=base_config, init_params=True)

    check_distribution(module, 0.0, 0.02, ignore_params=["ff_out", "ln_f", "attn_norm", "ff_norm", "wte"])
    for i in range(n_layers):
        check_distribution(module.transformer.blocks[i].ff_out, 0.00, 0.02 / math.sqrt(2 * n_layers))
        check_distribution(module.transformer.blocks[i].attn_norm, 1.00, 0.00)
        check_distribution(module.transformer.blocks[i].ff_norm, 1.00, 0.00)
    check_distribution(module.transformer.ln_f, 1.00, 0.00)
    check_distribution(module.transformer.ff_out, 0.00, 0.02)
    check_distribution(module.transformer.wte, 0.0, 0.02*0.5 * math.sqrt(d_model))