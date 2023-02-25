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

Example of commands:

```bash
    python scripts/lucas/s2ag/process_text.py\
        src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc \
        dst=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/stats/s2orc \
        cpu_count=32
    python scripts/lucas/s2ag/process_text.py\
        src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2ag \
        dst=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/stats/s2ag \
        cpu_count=32
```

On a Cirrascale machine, the step above takes about 15 minutes to run for
s2orc and 5 minutes for s2arg.

Output counts:

- Number of whitespace-separated tokens: 70.7B (s2orc), 15.5B (s2arg)
- Number of documents: 9.9M (s2orc, full-text papers), 91.1M (s2arg, abstracts)


TODOs:

- better management of dependencies


### Step 3: Load the data back into Athena

Run `export/load.sql` to load the data back into Athena.
Then, run `export/save.sql` to run the filters and export the data to S3.

## 2023-02-25: First Export

#### Data Info

- Corpus is located at `s3://ai2-s2-research-public/lucas/s2orc_oa_2022_01_03`
- It is comprised of 30 gzipped JSONL files.
- Each line is a JSON object with the following fields:
  - `id`: the corpus ID of the paper in Semantic Scholar. If you want to look up the paper, use `https://api.semanticscholar.org/CorpusID:<id>`
  - `text`: the text of the paper. Sections are separated by double newlines, i.e. `\n\n`


#### Filters

- language is `en` as identified by pycld3
- number of whitespace-separated tokens is at least 50
    - abstracts below 50 are typically parsing errors.
- number of whitespace-separated tokens is at most 50,000
    - past 50k, you typically have large books, vocabulary, number heavy reports, etc. Not worth it.
- the most frequent token matches the regex `^[A-Za-z][a-z]+$`
    - documents that have parsing errors or are number heavy usually have a non alpha token as the most frequent, e.g. `.` or `\n`.
- for documents that have at least 500 tokens, the most frequent token is at most 7.5% of the total number of tokens.
    - estimate for English put frequency of top word in a document at 5-10% of the total number of tokens. splitting differences and going with 7.5%.
- for documents that are less than 500 tokens, the most frequent token is at most 30% of the total number of tokens.
    - for shorter documents, frequency estimates from above are not as reliable. going for a more generous 30%.

#### Stats

- Number of whitespace-separated tokens: 72,582,009,602
- Number of documents: 74,772,626
