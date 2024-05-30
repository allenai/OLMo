import tempfile

import pytest
import torch

from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from olmo.model import OLMo


def test_olmo_model(model_path: str):
    model = OLMo.from_checkpoint(model_path)
    hf_model = OLMoForCausalLM.from_pretrained(model_path)

    tokenizer = OLMoTokenizerFast.from_pretrained(model_path)
    input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(input).unsqueeze(0)

    output = model(input_tensor)
    hf_output = hf_model(input_tensor)

    torch.testing.assert_allclose(output.logits, hf_output.logits)


def test_save_pretrained(model_path: str):
    tokenizer = OLMoTokenizerFast.from_pretrained(model_path)
    input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(input).unsqueeze(0)

    hf_model = OLMoForCausalLM.from_pretrained(model_path)
    hf_output = hf_model(input_tensor)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_model.save_pretrained(tmp_dir)

        saved_hf_model = OLMoForCausalLM.from_pretrained(tmp_dir)
        saved_hf_output = saved_hf_model(input_tensor)

        torch.testing.assert_allclose(saved_hf_output.logits, hf_output.logits)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices")
def test_auto_device_map_load(model_path: str):
    hf_model = OLMoForCausalLM.from_pretrained(model_path, device_map="auto")
    assert hf_model.device.type == "cuda"
