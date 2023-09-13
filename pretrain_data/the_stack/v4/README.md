Author: Akshita Bhagia @akshitab

# Version: v4

* Starting from v3.
* Removed documents matching the following criteria (Reference: StarCoder filtering heuristics):
        * Contains XML template code
        * HTML code-to-text ratio <= 0.2
        * Java, Python, JavaScript code-to-comment ratio <= 0.01 and > 0.8


## Steps to reproduce


### Run taggers on v3

```bash
../v3/run_starcoder_taggers.sh
```

### Create mixer configs

```bash
mkdir configs
./create_v4_configs.sh lang_list.txt configs
```

### Run mixer

```bash
./create_v4.sh lang_list.txt configs $MIXER_PATH
```
