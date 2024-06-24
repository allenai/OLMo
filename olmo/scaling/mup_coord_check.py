import argparse
import os
import time

from typing import List, Optional
import numpy as np
import torch
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes
from torch.utils.data import DataLoader

from olmo.config import ModelConfig, TrainConfig
from olmo.scaling.new_coord_check import (
    get_coord_data,
    plot_coord_data,
)

from olmo.data import build_train_dataloader

from olmo.model import OLMo
from olmo.torch_util import seed_all
from olmo.train import cross_entropy_loss


def load_mu_model(config: ModelConfig):
    config.use_mup = True
    model = OLMo(config, init_params=False)
    return model


def get_dataloader(cfg: TrainConfig, batch_size: int) -> DataLoader:
    # Set seed.
    seed_all(cfg.seed)

    cfg.global_train_batch_size = batch_size
    cfg.device_train_batch_size = batch_size // 1  # TODO: assuming single GPU for now
    train_loader = build_train_dataloader(cfg)
    return train_loader


def coord_check(mup, lr, optimizer, batch_size, nsteps, nseeds, args, plotdir="", legend=False):
    def model_generator(d_model, standparam=False):
        def f():
            config = ModelConfig.load(args.config_path, key="model")
            config.d_model = d_model
            model = load_mu_model(config)  # .to(args.device)

            if standparam:
                set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, "load_base_shapes needs to be nonempty"
                set_base_shapes(model, args.load_base_shapes)

            model.reset_parameters()  # to apply mup init
            return model

        return f

    optimizer = optimizer.replace("mu", "")
    widths = 2 ** np.arange(7, 14)
    # widths = 2 ** np.arange(7, 9)
    # widths = 2 ** np.arange(6, 8)
    models = {width: model_generator(width, standparam=not mup) for width in widths}

    train_config = TrainConfig.load(args.config_path)
    data_loader = get_dataloader(train_config, batch_size=batch_size)

    df = get_coord_data(
        models,
        data_loader,
        mup=mup,
        lr=lr,
        optimizer=optimizer,
        dict_in_out=True,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn=cross_entropy_loss,
        cuda=args.cuda,
        compute_z_loss=train_config.softmax_auxiliary_loss,
        show_progress=True,
    )

    prm = "mup" if mup else "sp"
    coords_file = os.path.join(plotdir, f"{prm}_olmo_{optimizer}_coord.csv")
    df.to_csv(coords_file, index=False)
    return plot_coord_data(
        df,
        legend=legend,
        save_to=os.path.join(plotdir, f"{prm}_olmo_{optimizer}_coord.png"),
        suptitle=f"{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
    )

def save_base_shapes(config_path: str, output_path: str, dims_to_scale: Optional[List] = None):
    if dims_to_scale is None:
        dims_to_scale = ["d_model"]

    print(f"saving base shapes at {output_path}")

    config = ModelConfig.load(config_path, key="model")
    base_shapes = get_shapes(load_mu_model(config))

    # just need to change whatever dimension(s) we are scaling
    # currently only scaling width, but may scale depth also
    # width scaling by d_model, but can also be done based on num_heads, etc.

    for dim in dims_to_scale:
        setattr(config, dim, getattr(config, dim) * 2)

    delta_shapes = get_shapes(load_mu_model(config))
    make_base_shapes(base_shapes, delta_shapes, savefile=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run coord check for OLMo model with μP",
    )

    parser.add_argument("config_path")

    parser.add_argument("--save_base_shapes", type=str, default="", help="file location to save base shapes at")
    parser.add_argument("--load_base_shapes", type=str, default="", help="file location to load base shapes from")

    parser.add_argument("--batch_size", type=int, default=20, metavar="N", help="batch size")

    parser.add_argument("--cuda", action="store_true", help="use CUDA")

    parser.add_argument(
        "--coord_check",
        action="store_true",
        help="test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.",
    )
    parser.add_argument("--coord_check_nsteps", type=int, default=3, help="Do coord check with this many steps.")
    parser.add_argument(
        "--coord_check_nseeds",
        type=int,
        default=3,
        help="number of seeds for testing correctness of μ parametrization",
    )

    parser.add_argument("--coord_check_save_path", type=str, default="coord_checks", help="dir location for saving coord check plots")

    args = parser.parse_args()
    print(args)

    if args.save_base_shapes:
        save_base_shapes(args.config_path, args.save_base_shapes)
        print("done and exit")
        import sys

        sys.exit()

    train_config = TrainConfig.load(args.config_path)
    data_loader = get_dataloader(train_config, batch_size=args.batch_size)

    if args.coord_check:
        print("testing parametrization")
        import os

        os.makedirs(args.coord_check_save_path, exist_ok=True)

        for use_mup in [True, False]:
            coord_check(
                mup=use_mup,
                lr=train_config.optimizer.learning_rate,
                optimizer=train_config.optimizer.name,
                batch_size=args.batch_size,
                nsteps=args.coord_check_nsteps,
                nseeds=args.coord_check_nseeds,
                args=args,
                plotdir=args.coord_check_save_path,
                legend=False,
            )
