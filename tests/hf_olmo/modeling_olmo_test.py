import tempfile

import pytest
import torch

from olmo.model import OLMo


def test_olmo_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast  # noqa: F401

    model = OLMo.from_checkpoint(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(input).unsqueeze(0)

    output = model(input_tensor)
    hf_output = hf_model(input_tensor)

    torch.testing.assert_close(hf_output.logits, output.logits)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices")
def test_flash_attention_2(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import hf_olmo  # noqa: F401

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_model_flash_attn = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(encoded_input).unsqueeze(0)

    hf_output = hf_model(input_tensor)
    hf_output_flash_attn = hf_model_flash_attn(input_tensor)

    torch.testing.assert_close(hf_output_flash_attn.logits, hf_output.logits)


def test_sdpa(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import hf_olmo  # noqa: F401

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_model_sdpa = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="sdpa")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(encoded_input).unsqueeze(0)

    hf_output = hf_model(input_tensor)
    hf_output_sdpa = hf_model_sdpa(input_tensor)

    torch.testing.assert_close(hf_output_sdpa.logits, hf_output.logits)


def test_gradient_checkpointing(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

    import hf_olmo  # noqa: F401

    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(encoded_input).unsqueeze(0)

    hf_output_no_checkpointing = hf_model(input_tensor)

    hf_model.gradient_checkpointing_enable()

    hf_output_checkpointing = hf_model(input_tensor)

    torch.testing.assert_close(hf_output_checkpointing.logits, hf_output_no_checkpointing.logits)


def test_gradient_checkpointing_disable(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

    import hf_olmo  # noqa: F401

    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(encoded_input).unsqueeze(0)

    hf_output = hf_model(input_tensor)

    hf_model.gradient_checkpointing_enable()
    hf_model.gradient_checkpointing_disable()

    hf_output_after_disable = hf_model(input_tensor)

    torch.testing.assert_close(hf_output_after_disable.logits, hf_output.logits)


def test_save_pretrained(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast  # noqa: F401

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(input).unsqueeze(0)

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_output = hf_model(input_tensor)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_model.save_pretrained(tmp_dir)

        saved_hf_model = AutoModelForCausalLM.from_pretrained(tmp_dir)
        saved_hf_output = saved_hf_model(input_tensor)

        torch.testing.assert_allclose(saved_hf_output.logits, hf_output.logits)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices")
def test_auto_device_map_load(model_path: str):
    from transformers import AutoModelForCausalLM

    from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast  # noqa: F401

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    assert hf_model.device.type == "cuda"
