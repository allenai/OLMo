from typing import Iterable, List, Set

import pytest

from olmo.data import IterableDataset


def pack(values: Iterable[int]) -> List[List[int]]:
    return [[x] for x in values]


def unpack(dataset: IterableDataset) -> List[int]:
    return [x["input_ids"][0] for x in dataset]


def test_iterable_dataset_size():
    dataset = IterableDataset(pack(range(20)), world_size=2, rank=0, shuffle=False)
    assert dataset.total_size == 20
    assert unpack(dataset) == list(range(0, 20, 2))

    dataset = IterableDataset(pack(range(20)), world_size=3, rank=0, shuffle=False, drop_last=False)
    assert dataset.total_size == 21
    assert unpack(dataset) == list(range(0, 20, 3))

    dataset = IterableDataset(pack(range(20)), world_size=3, rank=2, shuffle=False, drop_last=False)
    assert unpack(dataset) == list(range(2, 18, 3)) + [0]

    dataset = IterableDataset(pack(range(20)), world_size=3, rank=0, shuffle=False, drop_last=True)
    assert dataset.total_size == 18
    assert unpack(dataset) == list(range(0, 18, 3))


def test_iterable_dataset_max_examples():
    device_batch_size = 2
    dataset = IterableDataset(
        pack(range(20)), world_size=2, rank=0, shuffle=False, max_examples=2 * device_batch_size * 3
    )
    assert unpack(dataset) == [0, 2, 4, 6, 8, 10]


def test_iterable_dataset_start_step():
    device_batch_size = 2
    dataset = IterableDataset(
        pack(range(20)), world_size=2, rank=0, shuffle=False, start_index=2 * device_batch_size * 3
    )
    assert unpack(dataset) == [12, 14, 16, 18]


@pytest.mark.parametrize("world_size", [2, 4])
def test_iterable_dataset_restart_different_world_size(world_size: int):
    start_index = 4
    all_indices: Set[int] = set()
    for rank in range(world_size):
        dataset = IterableDataset(
            pack(range(20)), world_size=world_size, rank=rank, shuffle=False, start_index=start_index
        )
        indices = set(unpack(dataset))
        assert len(all_indices & indices) == 0
        all_indices.update(indices)
    assert all_indices == set(range(4, 20))
