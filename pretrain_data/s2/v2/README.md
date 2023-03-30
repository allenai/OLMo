# S2 V2

## S2ORC (Full-text Papers)

Cleaned version of the S2ORC corpus, which contains full-text papers across multiple disciplines.
Unflitered, the corpus contains 11.3M papers and 46.9B whitespace-separated tokens.
To clean S2ORC, we impose the following constraints:

- The paper must have a title and abstract.
- From each paper, we use [Grobid](https://github.com/kermitt2/grobid) to extract section headers and paragraphs; figures, tables, and references, and any other non-textual content is removed. Title and abstracts are also available, but they come from the S2 metadata, not Grobid.
- The paper must be in English.
  - To calculate the language, we use the [pycld3](https://github.com/bsolomon1124/pycld3) library
  - We run pycld3 on the first 2000 characters of each paragraph in the paper.
  - The language of the paper is the most common language of the paragraphs.
- The paper must have at least 500 words.
- The paper was published after 1969.
- The paper must have at least 5 paragraphs.
  - All sections that have a average log word probability of less than `-20` are removed.
  - To calculate the average log word probability, we use word frequencies extracted from the [1T Web Ngram corpus](https://catalog.ldc.upenn.edu/LDC2006T13); specifically, we use the list available [created by Rachel Tatman](https://www.kaggle.com/datasets/rtatman/english-word-frequency). A copy is hosted [here](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv).
- The most frequent word in the paper consists of alpha characters only, and it appears in less than 7.5% of the document.
  - Words are obtained by splitting the text on whitespace.


Train data is anything published before 2022-12-01; validation data is anything published after 2022-12-01 and until 2023-01-03.

|Split|Documents|Approx Word Count|Location|
|---|---|---|---|
train|8,855,051|39,088,566,059|`s3://ai2-llm/pretraining-data/sources/s2/v2_dedup/dataset=s2orc/split=train`
validation|83,217|465,425,062|`s3://ai2-llm/pretraining-data/sources/s2/v2_dedup/dataset=s2orc/split=valid`

The set above is deduped by paper ID only, meaning that we keep multiple version of a paper if we have any. If you want to keep only one version of each paper, the stats are:

|Split|Abstracts|Approx Word Count|Location|
|---|---|---|---|
train|8,207,327|35,933,376,971|`s3://ai2-llm/pretraining-data/sources/s2/v2_hard_dedup/dataset=s2orc/split=train`
validation|70,641|380,402,164|`s3://ai2-llm/pretraining-data/sources/s2/v2_hard_dedup/dataset=s2orc/split=valid`

## S2AG (Titles and Abstracts Papers)

The S2AG corpus contains titles and abstracts of papers in Semantic Scholar.
Unfiltered, the corpus contains 91.1M papers and 15.5B whitespace-separated tokens, but we impose the following constraints:

- Abstract must be in English.
  - To calculate the language, we once again use pycld3
- Title must be in English, or have average unigram log probability greater than -20.
- Abstract must be in English.
- Abstract must have higher than -20 average unigram log probability.
- Abstract must have at least 50 words.
- Abstract must have no more than 1000 words.
- The most frequent word in the union of text and abstract must be a 2+ character alpha word, or it can be `a` followed by a 2+ character alpha word.
- Paper was published after 1969.

|Split|Abstracts|Approx Word Count|Location|
|---|---|---|---|
train|59,161,485|10,963,156,902|`s3://ai2-llm/pretraining-data/sources/s2/v2_dedup/dataset=s2ag/split=train`
validation|119,268|26,130,632|`s3://ai2-llm/pretraining-data/sources/s2/v2_dedup/dataset=s2ag/split=valid`


## Format

Data is available in JSONL format at the following locations:

```
s3://ai2-s2-research-public/lucas/s2_oa_pretrain_data/
|-- v2/
|   |-- s2ag/
|   |   |-- train/
|   |   |-- valid/
|   |-- s2orc/
|   |   |-- train/
|   |   |-- valid/
```

Each directory contains 30 gzipped files, each of which contains a JSONL file. Each line contains the following keys:
- `id`: The paper ID.
- `sha1` (optional): The SHA1 hash of the paper.
- `text`: The text of the paper. Sections are separated by two newlines, i.e. `\n\n`; paragraphs are separated by a single newline, i.e. `\n`.
  - For full text papers, each text looks like `[title]\n\n[abstract]\n\n[section header]\n[paragraph]\n\n[paragraph]\n\n[section header]\n\n[paragraph]\n\n[...]`
  - For titles and abstracts, each text looks like `[title]\n\n[abstract]`


## How do I get counts?

First, create the following athena table:

```sql
CREATE EXTERNAL TABLE `s2_v2_dedup` (
id int,
sha1 string,
text string
)
PARTITIONED BY (dataset string, split string)
ROW FORMAT serde 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/s2/v2_dedup/'
```

Then, set up partitions for the table:


```sql
MSCK REPAIR TABLE `s2_v2_dedup`
```

Finally, run the following query:

```sql
SELECT
  dataset,
  split,
  COUNT(cnt) as docs_count,
  SUM(cnt) as tokens_count
FROM (
    SELECT CARDINALITY(
        filter(
            REGEXP_SPLIT(text, '\s+'),
            x -> LENGTH(TRIM(x)) > 0
        )
    ) AS cnt,
    dataset,
    split
    FROM "temp_lucas"."s2_v2_dedup"
)
GROUP BY dataset, split
```

## Steps to follow to recreate data

### S2AG

1. Run `obtain_data/s2ag.sql` in Athena.
2. Run `process_s2ag.py` to add language ID and log probabilities.

```bash
python pretrain_data/s2/v2/process_s2ag.py \
  src=s3://ai2-llm/pretraining-data/sources/s2/raw/2023_01_03/s2ag/ \
  dst=s3://ai2-llm/pretraining-data/sources/s2/v0/documents/dataset=s2ag \
  cpu_count=64
```
3. Import data into athena with `load_/s2ag.sql`.

### S2ORC
