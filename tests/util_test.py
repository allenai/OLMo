import json
from pathlib import Path
from typing import Any, List

from datasets import Dataset, DatasetDict

from olmo import util


def test_dir_is_empty(tmp_path):
    # Should return true if dir doesn't exist, or exists but is empty.
    dir = tmp_path / "foo"
    assert not dir.exists()
    assert util.dir_is_empty(dir)
    dir.mkdir(parents=True)
    assert util.dir_is_empty(dir)

    # Should return false if dir contains anything, even hidden files.
    (dir / ".foo").touch()
    assert not util.dir_is_empty(dir)


def _create_and_store_test_hf_dataset(data: List[Any], dataset_path: Path):
    dataset_path.mkdir(parents=True, exist_ok=True)
    test_file_path = dataset_path / "test.json"
    with test_file_path.open("w") as f:
        json.dump(data, f)


def test_load_hf_dataset_gets_correct_data(tmp_path: Path):
    dataset_path = tmp_path / "test_dataset"
    cache_path = tmp_path / "cache"

    data = [{"foo": i} for i in range(10)]
    _create_and_store_test_hf_dataset(data, dataset_path)

    dataset = util.load_hf_dataset(str(dataset_path), name=None, split="test", datasets_cache_dir=str(cache_path))
    assert isinstance(dataset, (Dataset, DatasetDict))
    for i in range(10):
        assert dataset[i]["foo"] == i


def test_load_hf_dataset_caches_dataset(tmp_path: Path):
    dataset_path = tmp_path / "test_dataset"
    cache_path = tmp_path / "cache"

    data = [{"foo": i} for i in range(10)]
    _create_and_store_test_hf_dataset(data, dataset_path)

    dataset = util.load_hf_dataset(str(dataset_path), name=None, split="test", datasets_cache_dir=str(cache_path))
    assert isinstance(dataset, (Dataset, DatasetDict))
    assert dataset[0]["foo"] == 0

    # Overwrite dataset data and check that old data is loaded
    data = [{"bar": i} for i in range(10)]
    _create_and_store_test_hf_dataset(data, dataset_path)

    dataset = util.load_hf_dataset(str(dataset_path), name=None, split="test", datasets_cache_dir=str(cache_path))
    assert isinstance(dataset, (Dataset, DatasetDict))
    assert dataset[0]["foo"] == 0
