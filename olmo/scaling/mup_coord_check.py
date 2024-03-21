import argparse
import os
import numpy as np
import torch

from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes
from torch.utils.data import DataLoader

from olmo.config import ModelConfig, TrainConfig
from olmo.tokenizer import Tokenizer
from olmo.data import DataCollator, build_memmap_dataset, IterableDataset
from olmo.scaling.model import MuOLMo
from olmo.torch_util import seed_all


def set_precision(t, precision):
    if precision == 'half':
        # do nothing since this is handled by AMP
        return t
    elif precision == 'float':
        return t.float()
    elif precision == 'double':
        return t.double()
    else:
        raise ValueError(f'invalid precision string {args.precision}')


def load_mu_model(config: ModelConfig):
    model = MuOLMo(config, init_params=False)
    return model


def get_batch_inputs(train_config: TrainConfig, tokenizer: Tokenizer, device: torch.device):

    # TODO: change to real batches, more number of inputs.
    input1 = tokenizer.encode("My name is OLMo!")
    input2 = tokenizer.encode("I'm a delightful large open language model :)")
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }

    return batch_inputs


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


def coord_check(mup, lr, optimizer, batch_size, nsteps, nseeds, args, plotdir='', legend=False):

    # TODO: change to all parameters that need to be scaled.
    def gen(d_model, standparam=False):
        def f():
            config = ModelConfig.load(args.config_path, key="model")
            config.d_model = d_model
            model = load_mu_model(config)  # .to(args.device)

            print(model)

            model = set_precision(model, args.precision)
            if standparam:
                set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, 'load_base_shapes needs to be nonempty'
                set_base_shapes(model, args.load_base_shapes)
            return model

        return f

    optimizer = optimizer.replace('mu', '')
    widths = 2 ** np.arange(7, 14 if optimizer == 'sgd' else 12)
    models = {w: gen(w, standparam=not mup) for w in widths}

    train_config = TrainConfig.load(args.config_path)
    # tokenizer = Tokenizer.from_train_config(train_config)

    # # TODO: temporary; change to real data and batching
    # batches = [get_batch_inputs(train_config, tokenizer, device=torch.device("cpu"))] * batch_size
    data_loader = get_dataloader(train_config, batch_size=batch_size)
    df = get_coord_data(models, data_loader, mup=mup, lr=lr, optimizer=optimizer,
                        dict_in_out=True, nseeds=nseeds, nsteps=nsteps, lossfn='nll', cuda=False)

    prm = 'μP' if mup else 'SP'
    return plot_coord_data(df, legend=legend,
                           save_to=os.path.join(plotdir, f'{prm.lower()}_trsfmr_{optimizer}_coord.png'),
                           suptitle=f'{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}',
                           face_color='xkcd:light grey' if not mup else None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="OLMo model with μP",
    )

    parser.add_argument("config_path")

    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--bias', action='store_true',
                        help='use bias')
    parser.add_argument('--save_base_shapes', type=str, default='',
                        help='file location to save base shapes at')
    parser.add_argument('--load_base_shapes', type=str, default='',
                        help='file location to load base shapes from')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum')
    parser.add_argument('--output_mult', type=float, default=1,
                        help='output is multiplied by sqrt(output_mult/d_model)')
    parser.add_argument('--input_mult', type=float, default=1,
                        help='input is multiplied by sqrt(input_mult*d_model)')
    parser.add_argument('--attn_mult', type=float, default=1,
                        help='attn is multiplied by sqrt(attn_mult)/head_dim')

    parser.add_argument('--optimizer', default='musgd', choices=['sgd', 'musgd', 'adam', 'muadam'])
    parser.add_argument('--init_var', type=float, default=1,
                        help='weights are initialized with variance init_var/ninp')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--precision', type=str, default='float',
                        help='float | double | half')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='path to save the final model')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='path to resume training')
    parser.add_argument('--log_dir', type=str, default='.',
                        help='path to save logs')
    parser.add_argument('--coord_check', action='store_true',
                        help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
    parser.add_argument('--coord_check_nsteps', type=int, default=3,
                        help='Do coord check with this many steps.')
    parser.add_argument('--coord_check_nseeds', type=int, default=3,
                        help='number of seeds for testing correctness of μ parametrization')
    parser.add_argument('--deferred_init', action='store_true',
                        help='Skip instantiating the base and delta models for mup. Requires torchdistx.')

    args = parser.parse_args()

    print(args)

    if args.save_base_shapes:
        print(f'saving base shapes at {args.save_base_shapes}')

        config = ModelConfig.load(args.config_path, key="model")

        model = load_mu_model(config)

        base_shapes = get_shapes(load_mu_model(config))

        # just need to change whatever dimension(s) we are scaling
        config.d_model = config.d_model * 2
        delta_shapes = get_shapes(load_mu_model(config))
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        print('done and exit')
        import sys;
        sys.exit()

    if args.coord_check:
        print('testing parametrization')
        import os
        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        coord_check(mup=True, lr=args.lr, optimizer=args.optimizer, batch_size=args.batch_size, nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, args=args, plotdir=plotdir, legend=False)
        coord_check(mup=False, lr=args.lr, optimizer=args.optimizer, batch_size=args.batch_size, nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, args=args, plotdir=plotdir, legend=False)
        import sys; sys.exit()
