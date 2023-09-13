import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from olmo.config import BaseConfig, StrEnum, TrainConfig


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
