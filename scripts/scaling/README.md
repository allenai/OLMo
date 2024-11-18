
# Scaling predictions

`main` is a set of tasks containing a subset of 5-shot RC core OLMES tasks and MMLU var (0 to 5 shot).

## 1. Predict final loss

```
python scripts/scaling/final.py -c scripts/scaling/final/7b_full.json -o final_task_losses.png -k main
```

## 2. Predict task loss to accuracy

```
python scripts/scaling/task_loss_to_accuracy.py -c scripts/scaling/final/7b_full.json -o task_loss_to_accuracy.png -k main
```

## 3. Stacked predictions: combine step 1 and step 2

```
python scripts/scaling/stacked.py -c scripts/scaling/final/7b_full.json -o stacked.png -k main
```

```
python scripts/scaling/stacked.py -c scripts/scaling/final/7b_full.json -o stacked.png -k main --moving_avg 20 --skip_perc 0.1
```

