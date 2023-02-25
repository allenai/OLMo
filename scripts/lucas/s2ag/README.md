# S2AG Tables From Public Release

Cutoff date: 2023-01-03

## Reproduction steps

### Step 1: Obtain the data

Using Athena, run `obtain/s2arg.sql` and `obtain/s2orc.sql` to create parquet
files with abstract and full text of papers respectively. Each dump takes about
10-15 minutes to run.

TODOs:

- make this a configurable python script.

### Step 2: Run language identification and extraction of stats

Run `preprocess_text.py` on the two dumps above to (1) run pycld3 to identify
the language of each paper, and (2) extract the number of tokens (whitespace
splitted) and the most common 100 tokens in each paper.

Installation steps:

```bash
    sudo apt install protobuf-compiler
    pip install -r requirements.txt
```

Example of command:

```bash
    python scripts/lucas/s2ag/process_text.py\
        src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc \
        dst=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/stats/s2orc \
        cpu_count=60
```

On a Cirrascale machine, the step above takes about 15 minutes to run for
s2orc and 5 minutes for s2arg.

Output counts:

- Number of wh-sep tokens: 69.4B (s2orc)
- Number of papers: 9.7M (s2orc)


TODOs:

- better management of dependencies


### Step 3: Load the data back into Athena
