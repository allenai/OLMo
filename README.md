<div align="center">
  <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/>
  <br>
  <br>
  <h1>OLMo: Open Language Model</h1>
</div>

## Installation

```
pip install ai2-olmo
```

## Fine-tuning

To fine-tune an OLMo model you'll first need to prepare your dataset by tokenizing and saving it to a numpy memory-mapped array. See [`scripts/prepare_tulu_data.py`](./scripts/prepare_tulu_data.py) for an example with the Tulu V2 dataset, which can be easily modified for other datasets.

Next, prepare your training config. There are many examples in the [`configs/`](./configs) directory. Make sure the model parameters match up with the model your fine-tuning. To be safe you can always start from the config that comes with the model checkpoint.

Then launch the training job:

```
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config} \
    --data.paths=[{path_to_data}/input_ids.npy] \
    --data.label_mask_paths=[{path_to_data}/label_mask.npy] \
    --load_path={path_to_checkpoint} \
    --reset_trainer_state
```
