import os
import tempfile

import numpy as np
from mup.shape import load_base_shapes

from olmo.config import TrainConfig
from olmo.scaling.mup_coord_check import coord_check, save_base_shapes


def test_save_base_shapes():
    config_path = "test_fixtures/mup_train_tiny.yaml"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "base-shapes.bsh")
        save_base_shapes(config_path, output_path)

        base_shapes = load_base_shapes(output_path)
        assert isinstance(base_shapes, dict)


def test_coord_check():
    config_path = "test_fixtures/mup_train_tiny.yaml"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "base-shapes.bsh")
        save_base_shapes(config_path, output_path)

        train_config = TrainConfig.load(config_path)
        # train_config.data.paths = ["../../test_fixtures/sample-tokens.npy"]
        coord_check(
            mup=True,
            train_config=train_config,
            widths=2 ** np.arange(4, 6),
            batch_size=2,
            nsteps=1,
            nseeds=1,
            output_dir=os.path.join(temp_dir, "plots"),
            legend=False,
            load_base_shapes=output_path,
            cuda=False,
            plot=False,
        )

        os.path.exists(os.path.join(temp_dir, "plots", "mup_olmo_adamw_coord.csv"))
