import torch

from olmo.data.util import get_document_lengths


def test_get_cumulative_document_lengths():
    eos_token_id = 50279

    # Should work when the instance starts with EOS token.
    assert get_document_lengths(
        torch.tensor([eos_token_id, 3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5]), eos_token_id=eos_token_id
    ).tolist() == [1, 5, 3, 2]

    # Should work when the instance ends with EOS token.
    assert get_document_lengths(
        torch.tensor([3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5, eos_token_id]), eos_token_id=eos_token_id
    ).tolist() == [5, 3, 3]
