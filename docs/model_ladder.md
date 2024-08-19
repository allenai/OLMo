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