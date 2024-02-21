from typing import Optional

import numpy as np

from olmo.mm_data.iterable_dataset import MMIterableDataset
from tests.data.iterable_dataset_test import MockWorkerInfo


class MockSequenceIndex:
    def __init__(self, size):
        self.num_sequences = size

    def iter_blocks(self, start, end, block_size, block_step):
        on = start
        while on < end:
            for _ in range(block_size):
                yield on % self.num_sequences
                on += 1
                if on >= end:
                    return
            on += block_step


class MockReader:
    def read_ranges(self, seq, sequence_length, segment_ids):
        return seq


class PatchedDataset(MMIterableDataset):
    # Patched to yield sequence numbers instead of items

    def __init__(
            self,
            num_sequences: int,
            global_batch_size: int = None,
            start_index: int = 0,
            drop_last: bool = False,
            max_examples: Optional[int] = None,
            num_workers: int = None,
            worker_id: int = None,
            world_size: Optional[int] = 1,
            rank: Optional[int] = 0,
            num_threads: Optional[int] = 0,
    ):
        if global_batch_size is None:
            global_batch_size = world_size
        super().__init__(
            MockReader(), None, None, None,
            global_batch_size=global_batch_size,
            start_index=start_index,
            drop_last=drop_last,
            max_examples=max_examples,
            world_size=world_size,
            rank=rank,
            num_threads=num_threads
        )
        if num_workers is None:
            self.worker_info = None
        else:
            self.worker_info = MockWorkerInfo(worker_id, num_workers)
        self._index = MockSequenceIndex(num_sequences)

    def _init_for_seed(self, seed):
        pass


def test_bounds():
    out = list(PatchedDataset(12, start_index=1, max_examples=8))
    assert out == list(range(1, 8))

    out = list(PatchedDataset(19, start_index=7, max_examples=17))
    assert out == list(range(7, 17))

    out = list(PatchedDataset(13, world_size=2, drop_last=True, start_index=2, rank=0))
    assert out == list(range(2, 12, 2))

    out = list(PatchedDataset(13, world_size=2, drop_last=False, start_index=2, rank=1))
    assert out == (list(range(3, 13, 2)) + [0])

    out = list(PatchedDataset(1, world_size=4, drop_last=False, rank=1))
    assert out == [0]


def test_batch_splitting():
    out = list(PatchedDataset(12, global_batch_size=2, num_workers=3, worker_id=0))
    assert out == [0, 1, 6, 7]

    out = list(PatchedDataset(12, global_batch_size=2, num_workers=3, worker_id=1))
    assert out == [2, 3, 8, 9]

    out = list(PatchedDataset(10, global_batch_size=2, num_workers=3, worker_id=2))
    assert out == [4, 5]

    out = list(PatchedDataset(10, global_batch_size=2, num_workers=3, worker_id=0))
    assert out == [0, 1, 6, 7]

    out = list(PatchedDataset(16, global_batch_size=4, num_workers=2,
                              world_size=2, rank=0, worker_id=0))
    assert out == [0, 1, 8, 9]

    out = list(PatchedDataset(16, global_batch_size=2, num_workers=2,
                              world_size=2, rank=0, worker_id=0))
    assert out == [0, 4, 8, 12]

    out = list(PatchedDataset(16, global_batch_size=4, num_workers=2,
                              world_size=2, rank=1, worker_id=1))
    assert out == [6, 7, 14, 15]


def test_threading():
    for num_threads in [1, 2, 4]:
        out = list(PatchedDataset(100, num_threads=num_threads))
        assert out == list(range(100))


def _test_end_to_end(ds_size, batch_size, num_workers, num_devices, start=0):
    device_iterators = []
    device_batch_size = batch_size // num_devices
    for rank in range(num_devices):
        worker_iterators = []
        for worker_id in range(num_workers):
            worker_iterators.append(iter(PatchedDataset(
                ds_size, batch_size, start,
                world_size=num_devices, rank=rank,
                num_workers=num_workers, worker_id=worker_id
            )))

        def get_device_batch(_worker_its):
            while True:
                for it in _worker_its:
                    batch = []
                    for _ in range(device_batch_size):
                        batch.append(next(it))
                    yield batch

        device_iterators.append(get_device_batch(worker_iterators))

    on = start
    n_batches = ds_size // batch_size
    for batch_id in range(n_batches):
        global_batch = []
        for it in device_iterators:
            global_batch += next(it)
        assert sorted(global_batch) == list(range(on, on + len(global_batch)))
        on += len(global_batch)


def test_end_to_end():
    _test_end_to_end(24, batch_size=12, num_workers=2, num_devices=2)
    _test_end_to_end(24, batch_size=12, num_workers=1, num_devices=4)
    _test_end_to_end(24, batch_size=12, num_workers=6, num_devices=2)
    _test_end_to_end(48, batch_size=24, num_workers=4, num_devices=4)
    _test_end_to_end(32, batch_size=6, num_workers=3, num_devices=2, start=2)
