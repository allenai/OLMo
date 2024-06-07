import pytest
import torch
import transformers
from packaging import version

from olmo import BlockType, Tokenizer, TrainConfig
from olmo.data import DataCollator
from olmo.model import OLMo
from olmo.torch_util import seed_all


@pytest.mark.skipif(
    version.parse(transformers.__version__) >= version.parse("4.40.0"),
    reason="hf_olmo auto classes are not compatible with transformers >=v4.40.0",
)
def test_auto_hf_classes(model_path: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from hf_olmo import OLMoConfig, OLMoForCausalLM, OLMoTokenizerFast
    from hf_olmo.convert_olmo_to_hf import write_config, write_model, write_tokenizer

    # model_path is an OLMo checkpoint.
    # Creates HF-compatible config.json
    write_config(model_path)
    write_tokenizer(model_path)
    write_model(model_path)

    config = AutoConfig.from_pretrained(model_path)
    assert isinstance(config, OLMoConfig)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    assert isinstance(model, OLMoForCausalLM)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert isinstance(tokenizer, OLMoTokenizerFast)


@pytest.mark.parametrize(
    "alibi, rope, flash_attn, block_type, multi_query_attention, cuda, dtype",
    [
        pytest.param(
            True, False, False, BlockType.sequential, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"
        ),
        pytest.param(
            False, False, False, BlockType.sequential, False, False, torch.bfloat16, id="posit-emb-cpu-bf16"
        ),
        pytest.param(
            True, False, False, BlockType.sequential, False, False, torch.float32, id="alibi-emb-cpu-f32"
        ),
        pytest.param(
            False, False, False, BlockType.sequential, False, False, torch.float32, id="posit-emb-cpu-f32"
        ),
        pytest.param(
            False, True, False, BlockType.sequential, False, False, torch.bfloat16, id="rope-emb-cpu-bf16"
        ),
        pytest.param(False, True, False, BlockType.sequential, False, False, torch.float32, id="rope-emb-cpu-f32"),
        pytest.param(
            True,
            False,
            False,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="alibi-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            True,
            False,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="rope-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            False,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="posit-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            True,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="posit-emb-flash-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            True,
            BlockType.sequential,
            False,
            True,
            torch.float16,
            id="posit-emb-flash-cuda-f16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False, False, False, BlockType.sequential, True, False, torch.float32, id="posit-emb-mqattn-cpu-f32"
        ),
    ],
)
def test_forward(
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    alibi: bool,
    rope: bool,
    flash_attn: bool,
    block_type: BlockType,
    multi_query_attention: bool,
    cuda: bool,
    dtype: torch.dtype,
):
    from hf_olmo import OLMoConfig, OLMoForCausalLM

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    train_config.model.alibi = alibi
    train_config.model.rope = rope
    train_config.model.flash_attention = flash_attn
    if flash_attn:
        train_config.model.attention_dropout = 0.0
    train_config.model.block_type = block_type
    train_config.model.multi_query_attention = multi_query_attention
    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"

    use_amp = dtype in {torch.float16, torch.bfloat16}

    seed_all(1234)
    model = OLMo(train_config.model).eval()

    hf_config = OLMoConfig(**model.config.asdict())

    seed_all(1234)
    hf_model = OLMoForCausalLM(hf_config, init_params=True).eval()

    input1 = tokenizer.encode("My name is OLMo!")
    input2 = tokenizer.encode("I'm a delightful large open language model :)")
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }

    # Run forward pass.
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
        ):
            output1 = model(torch.tensor(input1, device=model.device).unsqueeze(0))
            output2 = model(torch.tensor(input2, device=model.device).unsqueeze(0))
            batch_output = model(**batch_inputs)

            hf_output1 = hf_model(torch.tensor(input1, device=model.device).unsqueeze(0))
            hf_output2 = hf_model(torch.tensor(input2, device=model.device).unsqueeze(0))
            hf_batch_output = hf_model(**batch_inputs)

    # Check that logits from individual inputs are equal to logits from batch.
    # With using half-precision types these might have some big differences in a small
    # percentage of the elements.
    atol = 1e-2 if use_amp else None
    rtol = 1e3 if use_amp else None
    torch.testing.assert_close(
        hf_output1.logits[0][: len(input1)], hf_batch_output.logits[0][: len(input1)], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        hf_output2.logits[0][: len(input2)], hf_batch_output.logits[1][: len(input2)], rtol=rtol, atol=atol
    )

    torch.testing.assert_close(hf_output1.logits, output1.logits)
    torch.testing.assert_close(hf_output2.logits, output2.logits)
    torch.testing.assert_close(hf_batch_output.logits, batch_output.logits)
