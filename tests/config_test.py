import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest import TestCase

import numpy

from olmo.config import BaseConfig, DataConfig, StrEnum, TrainConfig


@dataclass
class FakeConfig(BaseConfig):
    paths: List[str]
    env_var: str


def test_resolvers(tmp_path: Path):
    os.environ["FOO"] = "bar"
    with open(tmp_path / "config.yaml", "w") as f:
        f.writelines(["paths: ${path.glob:*.md}\n", "env_var: ${oc.env:FOO}\n"])
    config = FakeConfig.load(tmp_path / "config.yaml")
    assert config.env_var == "bar"


def test_str_enum():
    class Constants(StrEnum):
        foo = "foo"
        bar = "bar"

    assert "foo" == Constants.foo


def test_save_and_load(train_config: TrainConfig, tmp_path: Path):
    train_config.model.n_layers = 5
    save_path = tmp_path / "conf.yaml"

    train_config.save(save_path)
    assert save_path.is_file()

    loaded_train_config = TrainConfig.load(save_path)
    assert loaded_train_config == train_config

    loaded_train_config = TrainConfig.load(save_path, ["model.n_layers=2"])
    assert loaded_train_config != train_config
    assert loaded_train_config.model.n_layers == 2


def test_new():
    config = TrainConfig.new(seed=2)
    assert config.seed == 2


class TestDataConfig(TestCase):
    def test_data_config(self):
        data_config = DataConfig.new()
        self.assertEqual(data_config.memmap_dtype, "uint16")
        self.assertEqual(data_config.effective_memmap_dtype, numpy.uint16)

        data_config.memmap_dtype = "uint32"
        self.assertEqual(data_config.effective_memmap_dtype, numpy.uint32)

        data_config.memmap_dtype = "uint64"
        self.assertEqual(data_config.effective_memmap_dtype, numpy.uint64)

        data_config.memmap_dtype = "unknown"
        with self.assertRaises(TypeError):
            data_config.effective_memmap_dtype
