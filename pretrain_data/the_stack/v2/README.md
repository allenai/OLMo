Author: Akshita Bhagia @akshitab

# Version: v2

* Starting from v1.
* Removed documents matching the following criteria (Reference: RedPajama code filtering heuristics):
        * Maximum line length > 1000 characters
        * Average line length > 100 characters
        * Proportion of alphanumeric characters < 0.25
        * Ratio of alphabetical characters to number of tokens < 1.5
* Number of documents: 255,064,041
* Number of tokens (GPT-NeoX tokenizer): 368.9 B


## Steps to reproduce


### Run taggers on v1

```bash
../v1/run_rpj_taggers.sh
```

### Create mixer configs

```bash
mkdir configs
./create_v2_configs.sh lang_list.txt configs
```

### Run mixer

```bash
./create_v2.sh lang_list.txt configs $MIXER_PATH
```
