# PPT2 Experiments

To create a Beaker session with WEKA mounted:

```shell
# Create beaker session with WEKA mounted
beaker session create --budget ai2/allennlp --bare --mount weka://oe-training-default=/weka/oe-training-default
```

To launch training runs:

```shell
export GPUS=4

# To launch the control.
ppt2/scripts/launch.sh ppt2/configs/peteish1-control.yaml

# To launch phase 0 on shuffle Dyck.
# TODO: Read shuffle-dyck from Google Cloud
scripts/launch.sh ppt2/configs/peteish1-phase0.yaml

# Launch phase1 on checkpoint 500, which is step-matched.
# TODO: Read phase0 checkpoint 250/500 from WEKA
scripts/launch.sh ppt2/configs/peteish1-phase1.yaml
```

Observations from running:

* Beaker error messages are now more informative around why things don't launch. Budget issues, etc.
* The workspace `ai2/willm-ppt2` currently has max priority Normal.
* Note `ai2/oe-training` was renamed `ai2/oe-base`.
* Clusters: `ai2/titan-cirrascale` is B200s, `ai2/ceres-cirrascale`, `ai2/neptune-cirrascale`, `ai2/jupiter-cirrascale-2`, etc.