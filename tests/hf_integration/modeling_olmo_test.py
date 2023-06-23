import torch

from olmo.model import Olmo


def test_olmo_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hf_integration import OLMoForCausalLM  # noqa: F401

    model = Olmo.from_checkpoint(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input = tokenizer.encode("My name is OLMo!")
    input_tensor = torch.tensor(input).unsqueeze(0)

    output = model(input_tensor)
    hf_output = hf_model(input_tensor)

    torch.testing.assert_allclose(output.logits, hf_output.logits)
