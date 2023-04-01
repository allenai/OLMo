# S2 V2

## Clean-up Steps

### S2ORC (Full-text Papers)

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

### S2AG (Titles and Abstracts Papers)

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


## Dataset Statistics

### How To Compute Them

First, create the following athena table:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS  `llm_s2_v2` (
  id string,
  source string,
  version string,
  added string,
  created string,
  text string
)
PARTITIONED BY (dataset string, split string)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/s2/v2/documents'
TBLPROPERTIES (
  'classification'='json',
  'compressionType'='gzip'
)
```

Then, set up partitions for the table:


```sql
MSCK REPAIR TABLE `llm_s2_v2`
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
    FROM "llm_s2_v2"
)
GROUP BY dataset, split
ORDER BY dataset, split
```

### Statistics

Papers published before 2022-12-01 are used as training data, and papers published on or after 2022-12-01 are used for validation.

| Dataset | Split   | Docs        | Tokens         |
|:-------:|:-------:|:-----------:|:--------------:|
|s2orc    | train   | 8,242,162   | 36,088,195,908 |
|s2orc    | valid   | 51,323      | 255,139,074    |
|s2ag     | train   | 59,382,301  | 11,009,123,378 |
|s2ag     | valid   | 111,228     | 24,398,512     |

## How to Create the Dataset

### S2AG

1. Run `obtain_data/s2ag.sql` in Athena.
2. Run `process_s2ag.py` to add language ID and log probabilities.

```bash
python pretrain_data/s2/v2/process_s2ag.py \
  src=s3://ai2-llm/pretraining-data/sources/s2/raw/2023_01_03/s2ag/ \
  dst=s3://ai2-llm/pretraining-data/sources/s2/v0/documents/dataset=s2ag \
  parallel=96
```
3. Import data into athena with `load_as_table/s2ag.sql`.
4. Run queries in `process_corpus_dedup` and `process_corpus_hard_dedup` to create V2 and V2 hard deduped versions of the corpus.

### S2ORC

1. Run `obtain_data/s2orc_paragraphs.sql` in Athena.
2. Run `process_s2orc.py` to add language ID and log probabilities.

```bash
python pretrain_data/s2/v2/process_s2orc.py \
  src=s3://ai2-llm/pretraining-data/sources/s2/raw/2023_01_03/s2orc/ \
  dst=s3://ai2-llm/pretraining-data/sources/s2/v0/documents/dataset=s2orc \
  parallel=96
```
3. Import data into athena with `load_as_table/s2orc_paragraphs.sql`.
4. Run queries in `process_corpus_dedup` and `process_corpus_hard_dedup` to create V2 and V2 hard deduped versions of the corpus.
