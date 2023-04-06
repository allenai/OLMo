from typing import List

import pytest

from olmo.tokenizer import Tokenizer


@pytest.mark.parametrize("add_special_tokens", [pytest.param(x, id=f"specials={x}") for x in (True, False)])
def test_encode(tokenizer: Tokenizer, lorem_ipsum: str, add_special_tokens: bool):
    truncate_to = 16

    # Encode without truncation.
    full_input_ids = tokenizer.encode(lorem_ipsum, add_special_tokens=add_special_tokens)

    # Now enable truncation and check.
    tokenizer.truncate_to = truncate_to
    input_ids = tokenizer.encode(lorem_ipsum, add_special_tokens=add_special_tokens)
    assert len(input_ids) == truncate_to
    if add_special_tokens:
        assert input_ids[-1] == tokenizer.eos_token_id
        assert input_ids[:-1] == full_input_ids[: truncate_to - 1]
    else:
        assert input_ids[-1] != tokenizer.eos_token_id
        assert input_ids == full_input_ids[:truncate_to]


@pytest.mark.parametrize("add_special_tokens", [pytest.param(x, id=f"specials={x}") for x in (True, False)])
def test_encode_batch(tokenizer: Tokenizer, lorem_ipsum_docs: List[str], add_special_tokens: bool):
    truncate_to = 16

    # Encode without truncation.
    all_full_input_ids = tokenizer.encode_batch(lorem_ipsum_docs, add_special_tokens=add_special_tokens)

    # Now enable truncation and check.
    tokenizer.truncate_to = truncate_to
    all_input_ids = tokenizer.encode_batch(lorem_ipsum_docs, add_special_tokens=add_special_tokens)
    for input_ids, full_input_ids in zip(all_input_ids, all_full_input_ids):
        assert len(input_ids) == truncate_to
        if add_special_tokens:
            assert input_ids[-1] == tokenizer.eos_token_id
            assert input_ids[:-1] == full_input_ids[: truncate_to - 1]
        else:
            assert input_ids[-1] != tokenizer.eos_token_id
            assert input_ids == full_input_ids[:truncate_to]
