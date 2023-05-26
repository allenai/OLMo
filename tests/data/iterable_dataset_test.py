from olmo.data import IterableDataset


def test_iterable_dataset_size():
    dataset = IterableDataset(list(range(20)), world_size=2, rank=0, shuffle=False)
    assert dataset.total_size == 20
    assert list(dataset) == list(range(0, 20, 2))

    dataset = IterableDataset(list(range(20)), world_size=3, rank=0, shuffle=False, drop_last=False)
    assert dataset.total_size == 21
    assert list(dataset) == list(range(0, 20, 3))

    dataset = IterableDataset(list(range(20)), world_size=3, rank=2, shuffle=False, drop_last=False)
    assert list(dataset) == list(range(2, 18, 3)) + [0]

    dataset = IterableDataset(list(range(20)), world_size=3, rank=0, shuffle=False, drop_last=True)
    assert dataset.total_size == 18
    assert list(dataset) == list(range(0, 18, 3))


def test_iterable_dataset_max_steps():
    batch_size = 2
    dataset = IterableDataset(list(range(20)), world_size=2, rank=0, shuffle=False, max_steps=batch_size * 3)
    assert list(dataset) == [0, 2, 4, 6, 8, 10]


def test_iterable_dataset_start_step():
    batch_size = 2
    dataset = IterableDataset(list(range(20)), world_size=2, rank=0, shuffle=False, start_step=batch_size * 3)
    assert list(dataset) == [12, 14, 16, 18]
