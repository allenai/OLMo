# OLMo: Open Language Model

## Setup

After cloning this repository, first install the latest [PyTorch](https://pytorch.org) according the official instructions relevant to your environment. Then install the remaining dependencies and code base by running:

```
pip install -e .[dev]
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

```bash
gantry run \
  --workspace ai2/llm-testing \
  --task-name "OLMo-tiny-c4" \
  --priority "high" \
  --beaker-image olmo-torch2-gantry \
  --cluster ai2/general-cirrascale-a100-80g-ib \
  --gpus 8 \
  --nfs \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --allow-dirty \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c 'torchrun --nproc-per-node 8 scripts/train.py configs/c4-tiny.yaml --save_folder=/net/nfs.cirrascale/allennlp/llm-checkpoints/tmp --run_name=c4-tiny-test-run'
```

#### Running on LUMI via Slurm

First read our [LUMI](./LUMI.md) documentation, but submitting a new job essentially just boils down to running this:

```bash
sbatch scripts/c4-tiny-on-lumi.sh
```

### Restarting a training job from a checkpoint

To restart a training job from a previous checkpoint, add the argument `--load_path=/path/to/checkpoint_directory` and re-launch the training run using the same method.

The checkpoints for a run will be located in the run's `--save_folder`. They're always subdirectories of `save_folder` that look like `step1000` for sharded checkpoints or `step1000-unsharded` for unsharded checkpoints.
There are also symlinks for the latest checkpoints in the form of `latest` and `latest-unsharded` for sharded and unsharded checkpoints, respectively.

Sharded checkpoints are the default type of checkpoint that's saved during training since these are the fastest, but you can also save unsharded checkpoints by setting `--save_interval_unsharded [INT]`.

If you plan to restart a training run using a *different* world size, you can only restart from an *unsharded* checkpoint.
However, you can convert a sharded checkpoint into an unsharded checkpoint by launching the script [scripts/unshard_checkpoint.sh](./scripts/unshard_checkpoint.sh) in the same way you launched the training script. Note that this needs to be launched with the exact same world size as when the *sharded* checkpoint was saved.

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

## Finding official runs

We keep all of our runs in WandB under [the "ai2-llm" entity](https://wandb.ai/ai2-llm).
We don't store model checkpoints in WandB. Those are in GCS under `gs://allennlp-olmo/<wandb_run_path>`.

### Highlighted models

 * 300M parameters, ~70B tokens, a starter model that's not completely random: https://wandb.ai/ai2-llm/LLM-scripts/runs/ed5krfk9
