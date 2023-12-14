import tempfile

from olmo.tokenizer import Tokenizer


def test_olmo_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    from hf_olmo import OLMoTokenizerFast  # noqa: F401

    tok = Tokenizer.from_checkpoint(model_path)
    hf_tok = AutoTokenizer.from_pretrained(model_path)

    input_str = "Hello, this is a test!"

    # Note: our tokenizer adds eos token by default, HF doesn't.
    tokenized = tok.encode(input_str, add_special_tokens=False)
    hf_tokenized = hf_tok.encode(input_str)

    assert tokenized == hf_tokenized

    # tokenized = tok([input_str], return_tensors="pt", max_length=5, truncation=True)
    hf_tokenized = hf_tok([input_str], return_tensors="pt", max_length=5, truncation=True)

    print(hf_tokenized)


def test_save_pretrained(model_path: str):
    from transformers import AutoTokenizer

    from hf_olmo import OLMoTokenizerFast  # noqa: F401

    hf_tok = AutoTokenizer.from_pretrained(model_path)

    input_str = "Hello, this is a test!"

    # Note: our tokenizer adds eos token by default, HF doesn't.
    hf_tokenized = hf_tok.encode(input_str)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_tok.save_pretrained(tmp_dir)

        saved_hf_tok = AutoTokenizer.from_pretrained(tmp_dir)
        saved_hf_tokenized = saved_hf_tok.encode(input_str)

        assert hf_tokenized == saved_hf_tokenized
