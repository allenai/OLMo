from typing import List

from mup.coord_check import _record_coords

from olmo.train import get_labels


def get_batch_loss(model, batch, lossfn, compute_z_loss):
    # TODO: move and import instead
    outputs = model(input_ids=batch["input_ids"])
    logits = outputs.logits

    logits_for_loss = logits[..., :-1, :].contiguous()
    # shape: (batch_size * seq_len, vocab_size)
    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
    # shape: (batch_size, seq_len)
    labels = get_labels(batch)
    # shape: (batch_size * seq_len,)
    labels = labels.view(-1)
    ce_loss, z_loss = lossfn(
        logits_for_loss,
        labels,
        ignore_index=-100,
        reduction="mean",
        compute_z_loss=compute_z_loss,
    )

    if compute_z_loss:
        assert z_loss is not None
        loss = ce_loss + z_loss
    else:
        loss = ce_loss
    return loss


def _get_coord_data(
    models,
    dataloader,
    optcls,
    nsteps=3,
    lossfn="xent",
    filter_module_by_name=None,
    fix_data=True,
    cuda=True,
    nseeds=1,
    output_fdict=None,
    input_fdict=None,
    param_fdict=None,
    show_progress=True,
    compute_z_loss=False,
):
    """Inner method for `get_coord_data`.

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.

    Inputs:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optcls:
            a function so that `optcls(model)` gives an optimizer used to train
            the model.
        nsteps:
            number of steps to train the model
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).

    """
    coordinates: List = []
    if fix_data:
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps
    if show_progress:
        from tqdm import tqdm

        pbar = tqdm(total=nseeds * len(models))

    import torch

    # for i in range(nseeds):
    #     torch.manual_seed(i)
    #     for width, model in models.items():
    for width, model_ in models.items():
        for i in range(nseeds):
            torch.manual_seed(i)
            model = model_()
            model = model.train()
            if cuda:
                model = model.cuda()
            optimizer = optcls(model)
            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(
                        module.register_forward_hook(
                            _record_coords(
                                coordinates,
                                width,
                                name,
                                batch_idx,
                                output_fdict=output_fdict,
                                input_fdict=input_fdict,
                                param_fdict=param_fdict,
                            )
                        )
                    )
                if cuda:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.cuda()

                loss = get_batch_loss(model, batch, lossfn, compute_z_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()

                if batch_idx == nsteps:
                    break
            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    import pandas as pd

    return pd.DataFrame(coordinates)


def get_coord_data(
    models, dataloader, optimizer="sgd", lr=None, mup=True, filter_trainable_by_name=None, **kwargs
):
    """Get coord data for coord check.

    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record coordinate
    statistics specified by `output_fdict`, `input_fdict`, `param_fdict`. By
    default, only `l1` is computed for output activations of each module.

    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.

    Inputs:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'sgd'`.
        lr:
            learning rate. By default is 0.1 for `'sgd'` and 1e-3 for others.
        mup:
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        filter_trainable_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be trained.
        nsteps:
            number of steps to train the model
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).

    """
    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-3
    if mup:
        from mup.optim import MuAdam as Adam
        from mup.optim import MuAdamW as AdamW
        from mup.optim import MuSGD as SGD
    else:
        from torch.optim import SGD, Adam, AdamW

    def get_trainable(model):
        params = model.parameters()
        if filter_trainable_by_name is not None:
            params = []
            for name, p in model.named_parameters():
                if filter_trainable_by_name(name):
                    params.append(p)
        return params

    if optimizer == "sgd":
        optcls = lambda model: SGD(get_trainable(model), lr=lr)  # noqa: E731
    elif optimizer == "adam":
        optcls = lambda model: Adam(get_trainable(model), lr=lr)  # noqa: E731
    elif optimizer == "adamw":
        optcls = lambda model: AdamW(get_trainable(model), lr=lr)  # noqa: E731
    elif optimizer is None:
        raise ValueError("optimizer should be sgd|adam|adamw or a custom function")

    data = _get_coord_data(models, dataloader, optcls, **kwargs)
    data["optimizer"] = optimizer
    data["lr"] = lr
    return data
