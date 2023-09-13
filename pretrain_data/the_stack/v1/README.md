Author: Akshita Bhagia @akshitab

# Version: v1

* Removed the following language files: `assembly`, `csv`, `json`, `json5`, `jsonld`, `jsoniq`, `svg`
* Removed copyright statements in code files.


## Steps to reproduce

### Run taggers on v0

```bash
../v0/run_basic_taggers.sh
../v0/run_copyright_comments_tagger.sh
```

### Create mixer configs

```bash
mkdir configs
./create_v1_configs.sh lang_list.txt configs
```

### Run mixer

```bash
./create_v1.sh lang_list.txt configs $MIXER_PATH
```

