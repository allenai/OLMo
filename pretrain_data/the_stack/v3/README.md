Author: Akshita Bhagia @akshitab

# Version: v3

* Starting from v2.
* Removed PII:
        * EMAIL_ADDRESS
        * PHONE_NUMBER
        * IP_ADDRESS
* Number of documents: 255,064,041 (same as v2)


## Steps to reproduce


### Run taggers on v2

```bash
../v2/run_pii_taggers.sh
```

### Create mixer configs

```bash
mkdir configs
./create_v3_configs.sh lang_list.txt configs
```

### Run mixer

```bash
./create_v3.sh lang_list.txt configs $MIXER_PATH
```
