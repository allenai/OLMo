import shutil
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from olmo.config import TrainConfig
from olmo.data import build_train_dataloader
from olmo.model import OLMo
from olmo.optim import build_optimizer, build_scheduler
from olmo.train import Trainer, cross_entropy_loss, fused_loss_fn


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", (16, 64))
@pytest.mark.parametrize("seq_len", (57, 300))
@pytest.mark.parametrize("vocab_size", (100, 200))
@pytest.mark.parametrize("z_loss_multiplier", (1e-4, 1e-5))
def test_fused_loss(batch_size, seq_len, vocab_size, z_loss_multiplier):
    logits = torch.randn(batch_size * seq_len, vocab_size).cuda()
    labels = torch.randint(0, vocab_size, (batch_size * seq_len,)).cuda()

    loss, z_loss = cross_entropy_loss(logits, labels, compute_z_loss=True, z_loss_multiplier=z_loss_multiplier)
    f_loss, f_z_loss = fused_loss_fn(logits, labels, compute_z_loss=True, z_loss_multiplier=z_loss_multiplier)

    # Note: This is allowing for very big differences!
    assert_close(loss, f_loss, atol=1e-2, rtol=1e-3)
    assert_close(z_loss, f_z_loss, atol=1e-2, rtol=1e-3)


def _get_module_names(checkpoint_traces_folder: Path) -> List[str]:
    module_names = []
    for trace_file in checkpoint_traces_folder.iterdir():
        trace_file_name = trace_file.name
        if trace_file_name.endswith("_input.pt"):
            module_name = trace_file_name.removesuffix("_input.pt")
        elif trace_file_name.endswith("_output.pt"):
            module_name = trace_file_name.removesuffix("_output.pt")
        else:
            assert False, f"Cannot get parameter from trace file {trace_file_name}"

        module_names.append(module_name)

    return module_names


def _compare_module_output(
    original_traces_folder: Path,
    new_traces_folder: Path,
    module_name: str,
    *,
    include_non_tensor_outputs: bool = True,
):
    original_module_input_path = original_traces_folder / f"{module_name}_input.pt"
    original_module_output_path = original_traces_folder / f"{module_name}_output.pt"
    new_module_input_path = new_traces_folder / f"{module_name}_input.pt"
    new_module_output_path = new_traces_folder / f"{module_name}_output.pt"

    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    original_input = torch.load(str(original_module_input_path), map_location=map_location)
    new_input = torch.load(str(new_module_input_path), map_location=map_location)

    assert (
        original_input.dtype == new_input.dtype
    ), f"{module_name} input dtype is different for new model. Original {original_input.dtype}, new {new_input.dtype}"
    assert (
        original_input.shape == new_input.shape
    ), f"{module_name} input shape is different for new model. Original {original_input.shape}, new {new_input.shape}"
    if "wte" in module_name:
        mismatching_element_count = torch.sum(torch.logical_not(torch.eq(original_input, new_input)))
        assert mismatching_element_count == 0, f"Number of {module_name} mis-matching inputs: %d"
    assert_close(
        new_input, original_input, msg=lambda msg: f"{module_name} inputs are not sufficiently close.\n{msg}"
    )

    original_output = torch.load(str(original_module_output_path), map_location=map_location)
    new_output = torch.load(str(new_module_output_path), map_location=map_location)

    if isinstance(original_output, torch.Tensor):
        assert (
            original_output.dtype == new_output.dtype
        ), f"{module_name} output dtype is different for new model. Original {original_output.dtype}, new {new_output.dtype}"
        assert_close(
            new_output,
            original_output,
            msg=lambda msg: f"{module_name} outputs are not sufficiently close.\n{msg}",
        )
    elif include_non_tensor_outputs:
        pass
        # logger.info("%s outputs: %s %s", module_name, original_output, new_output)


def _compare_module_outputs(
    original_traces_folder: Path,
    new_traces_folder: Path,
    *,
    include_non_tensor_outputs: bool = True,
):
    original_modules = set(_get_module_names(original_traces_folder))
    new_modules = set(_get_module_names(new_traces_folder))

    original_only_modules = original_modules - new_modules
    assert len(original_only_modules) == 0, f"Found modules only in base model: {', '.join(original_only_modules)}"

    new_only_modules = new_modules - original_modules
    assert len(new_only_modules) == 0, f"Found modules only in new model: {', '.join(new_only_modules)}"

    common_modules = original_modules.intersection(new_modules)
    for module_name in sorted(common_modules, key=lambda mod_name: int(mod_name.split("_")[-1])):
        _compare_module_output(
            original_traces_folder,
            new_traces_folder,
            module_name,
            include_non_tensor_outputs=include_non_tensor_outputs,
        )


def _get_train_config(model_path: Path, save_folder: Path) -> TrainConfig:
    cfg = TrainConfig.load(model_path / "config.yaml")
    cfg.save_folder = str(save_folder)
    cfg.data.paths = ["test_fixtures/random_data.npy"]
    cfg.precision = "amp_bf16"
    cfg.device_train_batch_size = 1
    cfg.global_train_batch_size = 1
    cfg.save_interval = None
    cfg.save_interval_unsharded = 1000

    # Keep model small enough
    cfg.model.vocab_size = 100
    cfg.model.embedding_size = 128
    cfg.model.eos_token_id = 2
    cfg.model.pad_token_id = 3

    # Need to set these to 0 to get deterministic results
    cfg.model.attention_dropout = 0.0
    cfg.model.residual_dropout = 0.0
    cfg.model.embedding_dropout = 0.0

    return cfg


def _train_model(
    model_path: str,
    cfg: TrainConfig,
    *,
    cuda: bool = False,
    replace_existing_model: bool = False,
):
    device = torch.device("cuda") if cuda else torch.device("cpu")

    olmo_model = OLMo(cfg.model).to_empty(device=device)
    olmo_model.reset_parameters()
    dist_model = olmo_model

    optim = build_optimizer(cfg, dist_model)
    scheduler = build_scheduler(cfg)
    train_loader = build_train_dataloader(cfg)

    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        dist_model=dist_model,  # type: ignore
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=[],
        indices_file=None,
    ) as trainer:
        if replace_existing_model:
            # Save model and move *.pt files to right place
            trainer.save_unsharded_checkpoint()
            for path in (Path(cfg.save_folder) / "step0-unsharded/").glob("*.pt"):
                path.rename(Path(model_path) / path.name)

        trainer.restore_unsharded_checkpoint(model_path)
        trainer.fit()

    if replace_existing_model:
        # Replace existing trace files
        model_traces_path = Path(model_path) / "traces"
        if model_traces_path.is_dir():
            shutil.rmtree(model_traces_path)
        shutil.copytree(Path(cfg.save_folder) / "traces", model_traces_path)


@pytest.mark.parametrize(
    "cuda",
    [
        pytest.param(False),
        pytest.param(
            True,
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
    ],
)
def test_train_first_step_forward_unchanged(
    xtiny_model_path: str, tmp_path: Path, cuda: bool, update_test: bool = False
):
    """
    This test checks that the output of model forward of the 1st step has not changed (relative to an existing checkpoint).

    Set update_test to True if a non-backwards-compatible change is being intentionally made and so this test needs to be updated.
    """
    cfg = _get_train_config(Path(xtiny_model_path), tmp_path / "test_forward")
    cfg.module_outputs_save_steps = [1, 2]
    cfg.stop_at = 2

    if update_test:
        np.save("test_fixtures/random_data.npy", np.random.randint(0, cfg.model.vocab_size, 2 ** 16))

    _train_model(
        xtiny_model_path,
        cfg,
        cuda=cuda,
        replace_existing_model=update_test,
    )

    assert (Path(cfg.save_folder) / "traces/step1").is_dir()
    _compare_module_outputs(Path(xtiny_model_path) / "traces/step1", Path(cfg.save_folder) / "traces/step1")

    assert not update_test, "Test successfully updated, please disable update_test"


@pytest.mark.parametrize(
    "cuda",
    [
        pytest.param(False),
        pytest.param(
            True,
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
    ],
)
def test_train_second_step_forward_unchanged(
    xtiny_model_path: str, tmp_path: Path, cuda: bool, update_test: bool = False
):
    """
    This test checks that the output of model forward of the 2nd step has not changed (relative to an existing checkpoint).

    Set update_test to True if a non-backwards-compatible change is being intentionally made and so this test needs to be updated.
    """
    cfg = _get_train_config(Path(xtiny_model_path), tmp_path / "test_forward")
    cfg.module_outputs_save_steps = [1, 2]
    cfg.stop_at = 2

    if update_test:
        np.save("test_fixtures/random_data.npy", np.random.randint(0, cfg.model.vocab_size, 2 ** 16))

    _train_model(
        xtiny_model_path,
        cfg,
        cuda=cuda,
        replace_existing_model=update_test,
    )

    assert (Path(cfg.save_folder) / "traces/step2").is_dir()
    _compare_module_outputs(Path(xtiny_model_path) / "traces/step2", Path(cfg.save_folder) / "traces/step2")

    assert not update_test, "Test successfully updated, please disable update_test"
