import argparse
import os

import numpy as np
import torch
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes
from torch.utils.data import DataLoader

from olmo.config import ModelConfig, TrainConfig
from olmo.data import DataCollator, IterableDataset, build_memmap_dataset
from olmo.scaling.coord_check import get_coord_data, plot_coord_data
from olmo.scaling.model import MuOLMo
from olmo.tokenizer import Tokenizer
from olmo.torch_util import seed_all
from olmo.train import cross_entropy_loss


def set_precision(t, precision):
    if precision == "half":
        # do nothing since this is handled by AMP
        return t
    elif precision == "float":
        return t.float()
    elif precision == "double":
        return t.double()
    else:
        raise ValueError(f"invalid precision string {args.precision}")


def load_mu_model(config: ModelConfig):
    config.mup = True
    model = MuOLMo(config, init_params=False)
    return model


def get_dataloader(cfg: TrainConfig, batch_size: int) -> DataLoader:
    # Set seed.
    seed_all(cfg.seed)

    # # Set some additional settings
    # if cfg.device_train_batch_size is None:
    #     log.warning(
    #         "device_train_batch_size is not set, so we're assuming we're running on 8 GPUs. "
    #         "Set that value on the command line if this is not true."
    #     )
    #     cfg.device_train_batch_size = cfg.global_train_batch_size // 8

    cfg.global_train_batch_size = batch_size
    cfg.device_train_batch_size = batch_size // 1  # TODO: assuming single GPU for now

    # Construct data loader.
    collator = DataCollator(pad_direction=cfg.data.pad_direction, pad_token_id=cfg.model.pad_token_id)
    dataset = build_memmap_dataset(cfg, cfg.data, include_instance_metadata=False)
    seed = cfg.data.seed if cfg.data.seed is not None else cfg.seed
    train_loader = DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            cfg.global_train_batch_size,
            seed=seed + (cfg.epoch or 0),
            shuffle=True,
            drop_last=cfg.data.drop_last,
            work_dir=None,
        ),
        batch_size=cfg.device_train_batch_size,
        drop_last=cfg.data.drop_last,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=None if cfg.data.num_workers == 0 else cfg.data.prefetch_factor,
        persistent_workers=False if cfg.data.num_workers == 0 else cfg.data.persistent_workers,
        timeout=cfg.data.timeout,
    )

    return train_loader


def coord_check(mup, lr, optimizer, batch_size, nsteps, nseeds, args, plotdir="", legend=False):
    # TODO: currently only for width; change to all parameters that need to be scaled.
    def gen(d_model, standparam=False):
        def f():
            config = ModelConfig.load(args.config_path, key="model")
            config.d_model = d_model
            model = load_mu_model(config)  # .to(args.device)

            model = set_precision(model, args.precision)
            if standparam:
                set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, "load_base_shapes needs to be nonempty"
                set_base_shapes(model, args.load_base_shapes)

            model.reset_parameters()  # to apply mup init TODO: confirm
            return model

        return f

    optimizer = optimizer.replace("mu", "")
    widths = 2 ** np.arange(7, 13)
    models = {w: gen(w, standparam=not mup) for w in widths}

    train_config = TrainConfig.load(args.config_path)
    # tokenizer = Tokenizer.from_train_config(train_config)

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

    prm = "μP" if mup else "SP"
    coords_file = "mup_coords.csv" if mup else "sp_coords.csv"
    df.to_csv(coords_file, index=False)
    return plot_coord_data(
        df,
        legend=legend,
        save_to=os.path.join(plotdir, f"{prm.lower()}_trsfmr_{optimizer}_coord.png"),
        suptitle=f"{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OLMo model with μP",
    )

    parser.add_argument("config_path")

    parser.add_argument("--save_base_shapes", type=str, default="", help="file location to save base shapes at")
    parser.add_argument("--load_base_shapes", type=str, default="", help="file location to load base shapes from")

    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument(
        "--output_mult", type=float, default=1, help="output is multiplied by sqrt(output_mult/d_model)"
    )
    parser.add_argument(
        "--input_mult", type=float, default=1, help="input is multiplied by sqrt(input_mult*d_model)"
    )
    parser.add_argument(
        "--attn_mult", type=float, default=1, help="attn is multiplied by sqrt(attn_mult)/head_dim"
    )

    parser.add_argument(
        "--optimizer", default="muadamw", choices=["sgd", "musgd", "adam", "muadam", "adamw", "muadamw"]
    )
    parser.add_argument(
        "--init_var", type=float, default=1, help="weights are initialized with variance init_var/ninp"
    )
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=20, metavar="N", help="batch size")

    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--precision", type=str, default="float", help="float | double | half")

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
        "--deferred_init",
        action="store_true",
        help="Skip instantiating the base and delta models for mup. Requires torchdistx.",
    )

    args = parser.parse_args()

    print(args)

    if args.save_base_shapes:
        print(f"saving base shapes at {args.save_base_shapes}")

        config = ModelConfig.load(args.config_path, key="model")

        model = load_mu_model(config)

        base_shapes = get_shapes(load_mu_model(config))

        # just need to change whatever dimension(s) we are scaling
        config.d_model = config.d_model * 2
        delta_shapes = get_shapes(load_mu_model(config))
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        print("done and exit")
        import sys

        sys.exit()

    if args.coord_check:
        print("testing parametrization")
        import os

        os.makedirs("coord_checks", exist_ok=True)
        plotdir = "coord_checks"
        coord_check(
            mup=True,
            lr=args.lr,
            optimizer=args.optimizer,
            batch_size=args.batch_size,
            nsteps=args.coord_check_nsteps,
            nseeds=args.coord_check_nseeds,
            args=args,
            plotdir=plotdir,
            legend=False,
        )
        coord_check(
            mup=False,
            lr=args.lr,
            optimizer=args.optimizer,
            batch_size=args.batch_size,
            nsteps=args.coord_check_nsteps,
            nseeds=args.coord_check_nseeds,
            args=args,
            plotdir=plotdir,
            legend=False,
        )
        import sys

        sys.exit()

    # TODO: train and eval muP models.
