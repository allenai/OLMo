import pytest
import torch
from torch.nn import CrossEntropyLoss

from dolma import DolmaGPT, ModelConfig, Tokenizer, TrainConfig
from dolma.data import DataCollator


@pytest.mark.parametrize(
    "alibi, flash_attn, cuda, dtype",
    [
        pytest.param(True, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(False, False, False, torch.bfloat16, id="posit-emb-cpu-bf16"),
        pytest.param(True, False, False, torch.float32, id="alibi-emb-cpu-f32"),
        pytest.param(False, False, False, torch.float32, id="posit-emb-cpu-f32"),
        pytest.param(
            True,
            False,
            True,
            torch.bfloat16,
            id="alibi-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
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
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
            ),
        ),
        pytest.param(
            True,
            True,
            True,
            torch.bfloat16,
            id="alibi-emb-flash-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
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
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
            ),
        ),
        pytest.param(
            True,
            True,
            True,
            torch.float16,
            id="alibi-emb-flash-cuda-f16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
            ),
        ),
        pytest.param(
            False,
            True,
            True,
            torch.float16,
            id="posit-emb-flash-cuda-f16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
            ),
        ),
    ],
)
def test_forward(
    train_config: TrainConfig, tokenizer: Tokenizer, alibi: bool, flash_attn: bool, cuda: bool, dtype
):
    torch.manual_seed(0)

    train_config.model.alibi = alibi
    train_config.model.flash_attention = flash_attn
    if flash_attn:
        train_config.model.attention_dropout = 0.0
    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"

    use_amp = dtype in {torch.float16, torch.bfloat16}

    model = DolmaGPT(train_config.model).eval()

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
    if not use_amp:
        torch.testing.assert_close(output1.logits[0][: len(input1)], batch_output.logits[0][: len(input1)])
        torch.testing.assert_close(output2.logits[0][: len(input2)], batch_output.logits[1][: len(input2)])


@pytest.mark.parametrize(
    "alibi, cuda, dtype",
    [
        pytest.param(True, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(False, False, torch.bfloat16, id="posit-emb-cpu-bf16"),
    ],
)
def test_backward(train_config: TrainConfig, tokenizer: Tokenizer, alibi: bool, cuda: bool, dtype):
    torch.manual_seed(0)

    use_amp = dtype in {torch.float16, torch.bfloat16}
    scaler = None if not (cuda and use_amp) else torch.cuda.amp.GradScaler()

    train_config.model.alibi = alibi
    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"

    model = DolmaGPT(train_config.model).train()

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


def test_configure_optimizer(model_config: ModelConfig):
    DolmaGPT(model_config).configure_optimizer()
