import pytest
import torch
from torch.nn import CrossEntropyLoss

from olmo import BlockType, Olmo, Tokenizer, TrainConfig
from olmo.config import PaddingDirection
from olmo.data import DataCollator


@pytest.mark.parametrize(
    "alibi, rope, flash_attn, block_type, multi_query_attention, cuda, dtype",
    [
        pytest.param(
            True, False, False, BlockType.sequential, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"
        ),
        pytest.param(
            True,
            False,
            False,
            BlockType.parallel,
            False,
            False,
            torch.bfloat16,
            id="alibi-emb-parallel-block-cpu-bf16",
        ),
        pytest.param(
            False, False, False, BlockType.sequential, False, False, torch.bfloat16, id="abs-emb-cpu-bf16"
        ),
        pytest.param(
            True, False, False, BlockType.sequential, False, False, torch.float32, id="alibi-emb-cpu-f32"
        ),
        pytest.param(False, False, False, BlockType.sequential, False, False, torch.float32, id="abs-emb-cpu-f32"),
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
            True,
            False,
            False,
            BlockType.parallel,
            False,
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
            id="abs-emb-cuda-bf16",
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
            id="abs-emb-flash-cuda-bf16",
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
            id="abs-emb-flash-cuda-f16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False, False, False, BlockType.sequential, True, False, torch.float32, id="abs-emb-mqattn-cpu-f32"
        ),
        pytest.param(
            False,
            False,
            False,
            BlockType.parallel,
            True,
            False,
            torch.float32,
            id="abs-emb-parallel-block-mqattn-cpu-f32",
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

    model = Olmo(train_config.model).eval()

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
            key_value_cache1 = model(
                torch.tensor(input1[:-1], device=model.device).unsqueeze(0), use_cache=True
            ).attn_key_values
            output1_from_cached = model(
                torch.tensor(input1[-1:], device=model.device).unsqueeze(0), past_key_values=key_value_cache1
            )
            output2 = model(torch.tensor(input2, device=model.device).unsqueeze(0))
            batch_output = model(**batch_inputs)
            batch_key_value_cache = model(
                batch_inputs["input_ids"][:, :-1],
                attention_mask=batch_inputs["attention_mask"][:, :-1],
                use_cache=True,
            ).attn_key_values
            batch_output_from_cached = model(
                batch_inputs["input_ids"][:, -1].unsqueeze(1),
                attention_mask=batch_inputs["attention_mask"],
                past_key_values=batch_key_value_cache,
            )

    # With using half-precision types these might have some big differences in a small
    # percentage of the elements.
    atol = 1e-2 if use_amp else None
    rtol = 1e3 if use_amp else None

    # Check that logits from individual inputs are equal to logits from batch.
    torch.testing.assert_close(
        output1.logits[0][: len(input1)], batch_output.logits[0][: len(input1)], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        output2.logits[0][: len(input2)], batch_output.logits[1][: len(input2)], rtol=rtol, atol=atol
    )

    # Check that output using cached attention keys + values matches.
    torch.testing.assert_close(output1.logits[0][-1], output1_from_cached.logits[0][-1], rtol=rtol, atol=atol)
    # For the batched output this only makes sense for the longer of the two inputs, since the shorter one is padded on the right.
    torch.testing.assert_close(output2.logits[0][-1], batch_output_from_cached.logits[1][-1], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "alibi, flash_attn, cuda, dtype",
    [
        pytest.param(True, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(False, False, False, torch.bfloat16, id="abs-emb-cpu-bf16"),
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
            id="abs-emb-cuda-bf16",
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
            id="abs-emb-flash-cuda-bf16",
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

    model = Olmo(train_config.model).train()

    with torch.autocast(
        device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
    ):
        # Forward pass to get logits.
        input_ids = torch.tensor(tokenizer.encode("My name is OLMo!"), device=model.device).unsqueeze(0)
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
            zeros = torch.zeros(parameter.size(), device=model.device)
            if (parameter.grad == zeros).all():
                raise RuntimeError(f"{name} has zero a gradient!")
        else:
            assert parameter.grad is None


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

    model = Olmo(train_config.model).eval()

    input1 = tokenizer.encode("My name is OLMo! ", add_special_tokens=False)
    input2 = tokenizer.encode("I'm a delightful large open language model :) ", add_special_tokens=False)
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }
    beam_search_kwargs = dict(beam_size=3, max_steps=5)

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
        ):
            output1 = model.generate(
                torch.tensor(input1, device=model.device).unsqueeze(0),  # type: ignore
                **beam_search_kwargs,
            )
            batch_output = model.generate(**{**batch_inputs, **beam_search_kwargs})

    torch.testing.assert_close(output1.scores[0], batch_output.scores[0])
