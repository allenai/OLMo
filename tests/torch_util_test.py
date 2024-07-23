import torch

from olmo import torch_util


def test_get_cumulative_document_lengths():
    assert torch_util.get_cumulative_document_lengths(
        torch.tensor(
            [
                [1, 5, 3, 2, 0],
                [5, 3, 3, 0, 0],
            ],
            dtype=torch.int32,
        )
    ).tolist() == [0, 1, 6, 9, 11, 16, 19, 22]
