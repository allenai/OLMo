# Model Ladder

The model ladder is a set of scripts that help you easily run models over a standardized set of parameter sizes and token multipliers.

## setup
You just probably only need beaker ganty

## example usage
For example this will train you a 150M model on the dolma17 data mix with a token multiplier of 20 * number of parameters (one chinchilla cuz who doesn't like more obscurity in naming) with a specifed run name and getting all the data from s3
```
scripts/beaker/ladder-launch.sh 1 --model 150M --data dolma17 --length 1xC --name testing-out-model-ladder --s3
```

## data mixes
Data mixes are defined in [named_data_mixes.py](olmo/data/named_data_mixes.py).

## detailed usage

### train command
```
usage: ladder.py train [-h] --model MODEL --data DATA [--length LENGTH] --name
                       NAME [--s3 | --no-s3] [--wandb | --no-wandb]
                       [--read_location READ_LOCATION]
                       [--write_location WRITE_LOCATION] [--save_overwrite]
                       [--load_path LOAD_PATH] [--eval_on_load]

options:
  -h, --help            show this help message and exit
  --model MODEL
  --data DATA
  --length LENGTH
  --name NAME
  --s3, --no-s3         read data from S3, write checkpoints to S3 (default:
                        False)
  --wandb, --no-wandb   create a run in wandb (default: True)
  --read_location READ_LOCATION
  --write_location WRITE_LOCATION
  --save_overwrite
  --load_path LOAD_PATH
  --eval_on_load
```