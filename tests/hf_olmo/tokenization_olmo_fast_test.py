from olmo.tokenizer import Tokenizer


def test_olmo_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    from hf_integration import OLMoTokenizerFast  # noqa: F401

    tok = Tokenizer.from_checkpoint(model_path)
    hf_tok = AutoTokenizer.from_pretrained(model_path)

    input_str = "Hello, this is a test!"

    # Note: our tokenizer adds eos token by default, HF doesn't.
    tokenized = tok.encode(input_str, add_special_tokens=False)
    hf_tokenized = hf_tok.encode(input_str)

    assert tokenized == hf_tokenized
