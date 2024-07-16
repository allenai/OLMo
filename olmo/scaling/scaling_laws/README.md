
## Scaling Laws

1. Download data from W&B

```commandline
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-mup/new_mup_olmo_128*' -y train/CrossEntropyLoss -o wandb_outputs/mup-olmo-128-train.csv
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-mup/new_mup_olmo_256*' -y train/CrossEntropyLoss -o wandb_outputs/mup-olmo-256-train.csv
```

2. Fit power law and make prediction by extrapolating over D (data size) on the same N (model size).

```commandline
python scripts/scaling/extrapolate_d.py -o olmo_tiny_scaling
```

3. Fit power law and make prediction by extrapolating over N (model size) on the same D (data size).

```commandline
# To pick a particular value of D and fit the power law on different models trained to D tokens
python scripts/scaling/extrapolate_n.py -o olmo_tiny_scaling -d 1000000000000
# To do the same thing for all values of D's, and predict a complete loss curve for the bigger model.
python scripts/scaling/extrapolate_n_forall_d.py -o olmo_tiny_scaling
```
