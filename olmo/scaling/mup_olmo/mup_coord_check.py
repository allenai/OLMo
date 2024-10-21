import argparse
import os
from typing import List, Optional

import numpy as np
from mup.coord_check import plot_coord_data
from torch.utils.data import DataLoader

from olmo.config import ModelConfig, TrainConfig
from olmo.data import build_train_dataloader
from olmo.scaling.mup_olmo.coord_check import get_coord_data
from olmo.scaling.mup_olmo.mup_utils import load_mu_model, save_base_shapes
from olmo.torch_util import seed_all
from olmo.train import cross_entropy_loss


def get_dataloader(cfg: TrainConfig, batch_size: int) -> DataLoader:
    # Set seed.
    seed_all(cfg.seed)

    cfg.global_train_batch_size = batch_size
    cfg.device_train_batch_size = batch_size // 1  # TODO: assuming single GPU for now
    train_loader = build_train_dataloader(cfg)
    return train_loader


def coord_check(
    mup: bool,
    widths: List,
    config_path: str,
    batch_size: int,
    nsteps: int,
    nseeds: int,
    cuda: bool = False,
    output_dir: str = "",
    load_base_shapes: Optional[str] = None,
    legend: str = "brief",
    plot: bool = True,
):
    def model_generator(d_model, standparam=False):
        def f():
            config = ModelConfig.load(config_path, key="model")
            config.d_model = d_model
            model = load_mu_model(config)

            if standparam:
                config.mup_base_shapes = None
            else:
                assert load_base_shapes, "load_base_shapes needs to be nonempty"
                config.mup_base_shapes = load_base_shapes

            model.set_base_shapes()
            model.reset_parameters()  # to apply mup init
            return model

        return f

    train_config = TrainConfig.load(config_path)
    optimizer = train_config.optimizer.name.replace("mu", "")
    lr = train_config.optimizer.learning_rate

    models = {width: model_generator(width, standparam=not mup) for width in widths}

    data_loader = get_dataloader(train_config, batch_size=batch_size)

    df = get_coord_data(
        models,
        data_loader,
        mup=mup,
        lr=lr,
        optimizer=optimizer,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn=cross_entropy_loss,
        cuda=cuda,
        compute_z_loss=train_config.softmax_auxiliary_loss,
        show_progress=True,
    )

    prm = "mup" if mup else "sp"
    os.makedirs(output_dir, exist_ok=True)
    coords_file = os.path.join(output_dir, f"{prm}_olmo_{optimizer}_coord.csv")
    df.to_csv(coords_file, index=False)
    if plot:
        # Plot no more than 20 graphs
        step_interval = max(nsteps // 20, 1)
        df = df[df["t"] % step_interval == 0]
        df.loc[:, "t"] /= step_interval

        plot_coord_data(
            df,
            legend=legend,
            save_to=os.path.join(output_dir, f"{prm}_olmo_{optimizer}_coord.png"),
            suptitle=f"{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}",
            face_color="xkcd:light grey" if not mup else None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run coord check for OLMo model with μP",
    )

    parser.add_argument("config_path")

    parser.add_argument("--save_base_shapes", type=str, default="", help="file location to save base shapes at")
    parser.add_argument("--load_base_shapes", type=str, default="", help="file location to load base shapes from")

    parser.add_argument("--batch_size", type=int, default=20, metavar="N", help="batch size")
    parser.add_argument("--widths", type=int, nargs="+", default=[2 ** i for i in range(5, 12)], help="widths to use for coord check")

    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--legend", type=str, help="'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.")

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

    parser.add_argument(
        "--coord_check_save_path",
        type=str,
        default="coord_checks",
        help="dir location for saving coord check plots",
    )

    args = parser.parse_args()
    print(args)

    if args.save_base_shapes:
        save_base_shapes(args.config_path, args.save_base_shapes)
        print("done and exit")
        import sys

        sys.exit()

    if args.coord_check:
        print("testing parametrization")
        import os

        os.makedirs(args.coord_check_save_path, exist_ok=True)

        for use_mup in [True, False]:
            coord_check(
                mup=use_mup,
                widths=args.widths,
                config_path=args.config_path,
                batch_size=args.batch_size,
                nsteps=args.coord_check_nsteps,
                nseeds=args.coord_check_nseeds,
                cuda=args.cuda,
                output_dir=args.coord_check_save_path,
                legend=args.legend,
                load_base_shapes=args.load_base_shapes,
            )
