import pytest
import torch
from torch.nn import CrossEntropyLoss

from dolma import Config, DolmaGPT, Tokenizer
from dolma.data import DataCollator, PaddingDirection


@pytest.mark.parametrize(
    "alibi, cuda",
    [
        pytest.param(True, False, id="alibi-emb-cpu"),
        pytest.param(False, False, id="posit-emb-cpu"),
        pytest.param(
            True,
            True,
            id="alibi-emb-cuda",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
            ),
        ),
        pytest.param(
            False,
            True,
            id="posit-emb-cuda",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
            ),
        ),
    ],
)
def test_forward(config: Config, tokenizer: Tokenizer, alibi: bool, cuda: bool):
    torch.manual_seed(0)

    config.alibi = alibi
    if cuda:
        config.init_device = "cuda"

    model = DolmaGPT(config).eval()

    input1 = tokenizer.encode("My name is DOLMA!")
    input2 = tokenizer.encode("I'm a delightful large open language model :)")
    batch_inputs = DataCollator(config=config, pad_direction=PaddingDirection.right)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=config.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }

    # Check that logits from individual inputs are equal to logits from batch.
    with torch.inference_mode():
        output1 = model(torch.tensor(input1).unsqueeze(0).to(device=config.device))
        output2 = model(torch.tensor(input2).unsqueeze(0).to(device=config.device))
        batch_output = model(**batch_inputs)

    torch.testing.assert_close(output1.logits[0][: len(input1)], batch_output.logits[0][: len(input1)])
    torch.testing.assert_close(output2.logits[0][: len(input2)], batch_output.logits[1][: len(input2)])


@pytest.mark.parametrize("alibi", [pytest.param(True, id="alibi-emb"), pytest.param(False, id="posit-emb")])
def test_backward(config: Config, tokenizer: Tokenizer, alibi: bool):
    torch.manual_seed(0)

    config.alibi = alibi
    model = DolmaGPT(config).train()

    # Forward pass to get logits.
    input_ids = torch.tensor(tokenizer.encode("My name is DOLMA!")).unsqueeze(0)
    logits = model(input_ids).logits

    # Compute loss.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Backward pass.
    loss.backward()

    # Check gradients.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            zeros = torch.zeros(parameter.size())
            if (parameter.grad == zeros).all():
                raise RuntimeError(f"{name} has zero a gradient!")
        else:
            assert parameter.grad is None


def test_configure_optimizer(config: Config):
    DolmaGPT(config).configure_optimizer()
