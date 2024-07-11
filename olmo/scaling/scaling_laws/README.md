
## Scaling Laws

1. Download data from W&B

```commandline
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-mup/new_mup_olmo_128*' -y train/CrossEntropyLoss -o wandb_outputs/mup-olmo-128-train.csv
python olmo/scaling/scaling_laws/download_wandb_logs.py -n 'ai2-llm/olmo-mup/new_mup_olmo_256*' -y train/CrossEntropyLoss -o wandb_outputs/mup-olmo-256-train.csv
```

2. Fit power law and make prediction for final loss.

```commandline
python scripts/scaling/predict_olmo_tiny_examples.py -o olmo_tiny_scaling
```
