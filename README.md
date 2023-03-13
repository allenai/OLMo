# DOLMA: Delightful Open Language Model from AI2

## Setup

After cloning this repository, first install the latest [PyTorch](https://pytorch.org) according the official instructions relevant to your environment. Then install the remaining dependencies and code base by running:

```
pip install -e .[dev] --config-settings editable_mode=compat
```

## Running experiments

### Using [beaker-gantry](https://github.com/allenai/beaker-gantry)

Train a model on c4 with gantry:

```bash
gantry run \
  --workspace ai2/llm-testing \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --venv base \
  --nfs \
  --priority preemptible \
  --gpus 8 \
  --beaker-image dolma-gantry \
  --cluster 'ai2/*-cirrascale' \
  --allow-dirty \
  -- composer scripts/train.py configs/1.2b-c4.yaml
```
