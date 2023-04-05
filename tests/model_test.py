import pytest
import torch
from torch.nn import CrossEntropyLoss

from dolma import BlockType, Dolma, ModelConfig, Tokenizer, TrainConfig
from dolma.composer import build_optimizer
from dolma.config import PaddingDirection
from dolma.data import DataCollator


@pytest.mark.parametrize(
    "alibi, rope, flash_attn, block_type, cuda, dtype",
    [
        pytest.param(True, False, False, BlockType.sequential, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(
            True, False, False, BlockType.parallel, False, torch.bfloat16, id="alibi-emb-parallel-block-cpu-bf16"
        ),
        pytest.param(False, False, False, BlockType.sequential, False, torch.bfloat16, id="posit-emb-cpu-bf16"),
        pytest.param(True, False, False, BlockType.sequential, False, torch.float32, id="alibi-emb-cpu-f32"),
        pytest.param(False, False, False, BlockType.sequential, False, torch.float32, id="posit-emb-cpu-f32"),
        pytest.param(False, True, False, BlockType.sequential, False, torch.bfloat16, id="rope-emb-cpu-bf16"),
        pytest.param(False, True, False, BlockType.sequential, False, torch.float32, id="rope-emb-cpu-f32"),
        pytest.param(
            True,
            False,
            False,
            BlockType.sequential,
            True,
            torch.bfloat16,
            id="alibi-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            True,
            False,
            False,
            BlockType.parallel,
            True,
            torch.bfloat16,
            id="alibi-emb-parallel-block-cuda-bf16",
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
            True,
            torch.float16,
            id="posit-emb-flash-cuda-f16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
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
    cuda: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    train_config.model.alibi = alibi
    train_config.model.rope = rope
    train_config.model.flash_attention = flash_attn
    if flash_attn:
        train_config.model.attention_dropout = 0.0
    train_config.model.block_type = block_type
    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"

    use_amp = dtype in {torch.float16, torch.bfloat16}

    model = Dolma(train_config.model).eval()

    input1 = tokenizer.encode("My name is DOLMA!")
    input2 = tokenizer.encode("I'm a delightful large open language model :)")
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=train_config.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }

    # Run forward pass.
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
        ):
            output1 = model(torch.tensor(input1, device=train_config.device).unsqueeze(0))
            output2 = model(torch.tensor(input2, device=train_config.device).unsqueeze(0))
            batch_output = model(**batch_inputs)

    # Check that logits from individual inputs are equal to logits from batch.
    # With using half-precision types these might have some big differences in a small
    # percentage of the elements.
    atol = 1e-2 if use_amp else None
    rtol = 1e3 if use_amp else None
    torch.testing.assert_close(
        output1.logits[0][: len(input1)], batch_output.logits[0][: len(input1)], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        output2.logits[0][: len(input2)], batch_output.logits[1][: len(input2)], rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "alibi, flash_attn, cuda, dtype",
    [
        pytest.param(True, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(False, False, False, torch.bfloat16, id="posit-emb-cpu-bf16"),
        pytest.param(
            True,
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
            True,
            True,
            torch.bfloat16,
            id="posit-emb-flash-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
                pytest.mark.skipif(
                    torch.cuda.device_count() < 1 or "A100" not in torch.cuda.get_device_name(),
                    reason="Requires A100 GPU type",
                ),
            ),
        ),
    ],
)
def test_backward(
    train_config: TrainConfig, tokenizer: Tokenizer, alibi: bool, flash_attn: bool, cuda: bool, dtype
):
    torch.manual_seed(0)

    use_amp = dtype in {torch.float16, torch.bfloat16}
    scaler = None if not (cuda and use_amp) else torch.cuda.amp.GradScaler()

    train_config.model.alibi = alibi
    train_config.model.flash_attention = flash_attn
    if flash_attn:
        train_config.model.attention_dropout = 0.0
    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"

    model = Dolma(train_config.model).train()

    with torch.autocast(
        device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
    ):
        # Forward pass to get logits.
        input_ids = torch.tensor(tokenizer.encode("My name is DOLMA!"), device=train_config.device).unsqueeze(0)
        logits = model(input_ids).logits

        # Compute loss.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Backward pass.
    if scaler is not None:
        scaler.scale(loss).backward()  # type: ignore
    else:
        loss.backward()

    # Check gradients.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            zeros = torch.zeros(parameter.size(), device=train_config.device)
            if (parameter.grad == zeros).all():
                raise RuntimeError(f"{name} has zero a gradient!")
        else:
            assert parameter.grad is None


def test_build_optimizer(model_config: ModelConfig):
    build_optimizer(Dolma(model_config))


@pytest.mark.parametrize(
    "cuda, dtype",
    [
        pytest.param(False, torch.float32, id="cpu-fp32"),
        pytest.param(
            True,
            torch.float32,
            id="cuda-fp32",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        # TODO: with an uninitialized model like we have here we'll end up with nan's
        # when we use half-precision. So eventually we should use a trained model in these tests.
        #  pytest.param(False, torch.bfloat16, id="cpu-bf16"),
    ],
)
def test_generate(
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    cuda: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    # Should always pad left when generating.
    train_config.data.pad_direction = PaddingDirection.left
    # We also need to use a relative positional embedding so that the
    # padding doesn't affect the results.
    train_config.model.alibi = True

    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"
    use_amp = dtype in {torch.float16, torch.bfloat16}

    model = Dolma(train_config.model).eval()

    input1 = tokenizer.encode("My name is DOLMA! ", add_special_tokens=False)
    input2 = tokenizer.encode("I'm a delightful large open language model :) ", add_special_tokens=False)
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=train_config.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }
    beam_search_kwargs = dict(beam_size=3, max_steps=5)

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
        ):
            output1 = model.generate(
                torch.tensor(input1, device=train_config.device).unsqueeze(0),  # type: ignore
                torch.tensor([[1.0] * len(input1)], device=train_config.device),
                **beam_search_kwargs,
            )
            batch_output = model.generate(**{**batch_inputs, **beam_search_kwargs})

    torch.testing.assert_close(output1.scores[0], batch_output.scores[0])
