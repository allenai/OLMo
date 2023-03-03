from pathlib import Path

from dolma.config import Config, TrainConfig


def test_save_and_load(tmp_path: Path):
    train_config = TrainConfig(model=Config(n_layers=5))
    save_path = tmp_path / "conf.yaml"

    train_config.save(save_path)
    assert save_path.is_file()

    loaded_train_config: TrainConfig = train_config.load(save_path)
    assert loaded_train_config == train_config

    loaded_train_config = train_config.load(save_path, ["model.n_layers=2"])
    assert loaded_train_config != train_config
    assert loaded_train_config.model.n_layers == 2
