# OLMo: Open Language Model

## Setup

After cloning this repository, first install the latest [PyTorch](https://pytorch.org) according the official instructions relevant to your environment. Then install the remaining dependencies and code base by running:

```
pip install -e .
```

## Running LM pre-training jobs

Our training script is [scripts/train.py](./scripts/train.py), which should be launched either through `torchrun` or Slurm (see below) since it only supports distributed training (on GPUs).
The first argument to the training script is a path to a [training configuration file](./configs/).
Then it takes any number of optional arguments that can be used to override values from the configuration file using dot notation.
For example, to change the learning rate you'd pass `--optimizer.learning_rate=0.0001`.

### Launching a training job

In the examples below we'll focus on training the "tiny" model on 8 GPUs and we'll assume that you've cloned this repository and are running all of the commands from the repository root,
whether that be on your laptop, on LUMI, or in a Beaker interactive session on Cirrascale.

#### Running on Cirrascale in a Beaker interactive session

```bash
run_name=c4-tiny-test-run
torchrun --nproc-per-node=8 scripts/train.py configs/c4-tiny.yaml \
  --run_name=${run_name} \
  --save_folder=/tmp/${run_name}  # change to somewhere permanent for a real run
```

#### Running on Cirrascale via [beaker-gantry](https://github.com/allenai/beaker-gantry)

Check the script at [`scripts/beaker/olmo-small-ablation-on-gantry.sh`](scripts/beaker/olmo-small-ablation-on-gantry.sh) for an example on how to run a training job on Cirrascale. Using that script, you can launch a training job like this:

```bash
CONFIG_PATH=configs/choose_a_config.yml \
LOAD_PATH=/optional/path/to/checkpoint/ \
  bash scripts/beaker/olmo-small-ablation-on-gantry.sh
```

If `CONFIG_PATH` is not specified, the default config is `configs/olmo-small-ablation.yaml`. If `LOAD_PATH` is not specified, the training will start from scratch.

#### Running on LUMI via Slurm

First read our [LUMI](docs/LUMI.md) documentation, but submitting a new job essentially just boils down to running this:

```bash
sbatch scripts/lumi/c4-small-on-lumi.sh
```

### Restarting a training job from a checkpoint

To restart a training job from a previous checkpoint, add the argument `--load_path=/path/to/checkpoint_directory` and re-launch the training run using the same method.

The checkpoints for a run will be located in the run's `--save_folder`. They're always subdirectories of `save_folder` that look like `step1000` for sharded checkpoints or `step1000-unsharded` for unsharded checkpoints.
There are also symlinks for the latest checkpoints in the form of `latest` and `latest-unsharded` for sharded and unsharded checkpoints, respectively.

Sharded checkpoints are the default type of checkpoint that's saved during training since these are the fastest, but you can also save unsharded checkpoints by setting `--save_interval_unsharded [INT]`.

If you plan to restart a training run using a *different* world size, you can only restart from an *unsharded* checkpoint.
However, you can convert a sharded checkpoint into an unsharded checkpoint by launching the script [scripts/unshard.sh](./scripts/unshard.sh) in the same way you launched the training script. Note that this needs to be launched with the exact same world size as when the *sharded* checkpoint was saved.

## Finding official runs and checkpoints

We track all of our runs in Weights & Biases under [the "ai2-llm" entity](https://wandb.ai/ai2-llm).
The corresponding checkpoints are stored in GCS under `gs://ai2-olmo/<wandb_run_path>`.
For example, checkpoints for the run [https://wandb.ai/ai2-llm/c4-small/runs/euox4j8q](https://wandb.ai/ai2-llm/c4-small/runs/euox4j8q) are located at [gs://ai2-olmo/ai2-llm/c4-small/euox4j8q/](https://console.cloud.google.com/storage/browser/ai2-olmo/ai2-llm/c4-small/euox4j8q).

You can load a checkpoint like this:

```python
from olmo import OLMo, Tokenizer

checkpoint = "gs://ai2-olmo/ai2-llm/c4-small/euox4j8q/step73000-unsharded"
model = OLMo.from_checkpoint(checkpoint, device="cuda")
tokenizer = Tokenizer.from_checkpoint(checkpoint)
```

### Highlighted checkpoints

 * `gs://ai2-olmo/ai2-llm/c4-small/euox4j8q/step73000-unsharded` - 1B parameters, 150B tokens, this one of our first decent checkpoints at the 1B scale.

## Generating text

You can use the `generate()` method to produce text using beam search with a variety of options.

For example:

```python
# Prepare inputs.
# Note: we don't want the EOS token added to the end of the input, hence
# the `add_special_tokens=False`.
input_ids = tokenizer.encode("I'm a large language model, ", add_special_tokens=False)
# `model.generate()` expects a batch.
input_tensor = torch.tensor(input_ids).unsqueeze(0)

# Run beam search.
outputs = model.generate(input_tensor, max_steps=3, beam_size=3)

# The output token IDs are shape (batch_size, beam_size, max_steps)
best_generation = outputs.token_ids[0][0].tolist()
print(tokenizer.decode(best_generation))
```

## Debugging

### Finding the cause of hangs

Hangs in distributed training can be due to several different causes, including
bad user code, AMD/Nvidia memory-allocation issues, or issues in hardware setup.
These issues can be difficult to root-cause and even harder to fix.

One approach we use to find the cause of a hang in distributed training is to first identify which processes/nodes are hanging. The [scripts/pyspy_all_processes.sh](https://github.com/allenai/OLMo/blob/main/scripts/pyspy_all_processes.sh) script retrieves the python state of relevant python processes using `pyspy`. A process/node with different state may be experiencing a hang.

If a hang is suspected to be in GPU code, then you can run `gcore <pid>` on a hanging process to get core dumps. Then you can run `gdb <corefile>` and check where the code is hanging from a C++ perspective. Code being stuck on a GPU memory allocation (malloc) may be indicative of a hardware/setup issue rather than a problem in training code.

### Comparing two models that should be identical

There are some scenarios when one might want to investigate why two models/setups that should be identical are yielding different results. A naive solution is to run both setups side-by-side and compare results manually (and this might not be possible if you have just 1 GPU).

An alternative for comparing OLMo models is to run the training of both models with the `--module_outputs_save_steps=[<list of steps]` config option. This causes OLMo to save a portion of the inputs & outputs of each OLMo submodule into a `traces/` folder at the model step's save location. Then [script/compare_module_outputs.py](https://github.com/allenai/OLMo/blob/main/scripts/compare_module_outputs.py) can be used to compare these portions of inputs & outputs, thus hopefully isolating the issue to a subset of model modules. See [script/compare_module_outputs.py](https://github.com/allenai/OLMo/blob/main/scripts/compare_module_outputs.py) for more details on its usage.

When comparing different hardware or dependency setups, it is possible that model
state gets corrupted before the first forward pass of training. One can check this
by running training with `--force_save_unsharded --dry_run --load_path=<original_model_path>` to save a checkpoint after the original model has loaded but before training has started. Then [scripts/compare_model_state.py](https://github.com/allenai/OLMo/blob/main/scripts/compare_model_state.py) can be used to see if parameters are different between the 2 models.