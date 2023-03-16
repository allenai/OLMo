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
  --env WANDB_ENTITY=ai2-llm \
  --env WANDB_PROJECT=testing \
  --venv base \
  --nfs \
  --priority preemptible \
  --gpus 8 \
  --beaker-image dolma-gantry \
  --cluster 'ai2/*-cirrascale' \
  --allow-dirty \
  -- composer scripts/train.py configs/1.2b-c4.yaml
```

Train the 70B model on c4 with gantry across multiple nodes:

```bash
gantry run \
  --workspace ai2/llm-testing \
  --priority "high" \
  --beaker-image dolma-gantry \
  --cluster ai2/general-cirrascale-a100-80g-ib \
  --gpus 8 \
  --nfs \
  --env WORLD_SIZE=32 \
  --env GPUS=8 \
  --env NCCL_DEBUG=INFO \
  --env SCRATCH_DIR=/tmp/scratch \
  --env FLASH_DIR=/tmp/flash \
  --env WANDB_PROJECT=dolma-beaker-ib \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --replicas 4 \
  --leader-selection \
  --host-networking \
  --allow-dirty \
  --venv base \
  --yes \
  -- /bin/bash -c 'composer --master_addr $BEAKER_LEADER_REPLICA_HOSTNAME --world_size $WORLD_SIZE --node_rank $BEAKER_REPLICA_RANK -n $GPUS --master_port 1234 scripts/train.py configs/70b-c4.yaml'
```

This may require a reservation on the Infiniband cluster.

See the [Beaker documentation](https://beaker-docs.apps.allenai.org/distributed-training.html) for more information on distributed training.
