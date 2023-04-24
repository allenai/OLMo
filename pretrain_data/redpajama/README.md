# RedPajama Dataset

This dataset is a a copy of [RedPajama][1] for [together.xyz][2] that has been converted in the format
necessary for LLM pre-training.

## Obtaining the Dataset

To obtain the dataset, run the `download.sh` script. This will download the dataset to `s3://ai2-llm/pretraining-data/sources/redpajama/raw/data/`, and keep track of which portions of the dataset have bee successfully downloaded at `s3://ai2-llm/pretraining-data/sources/redpajama/raw/metadata`.

## Converting and Creating Splits

To convert the dataset to the format required in training, we run the `v1.py` script. This will split data, reformat it, and place it at `s3://ai2-llm/pretraining-data/sources/redpajama/v1/documents/`.

To create the splits, we use the following logic:

1. Load each record file and discard any unicode errors: `json.loads(ln.encode("utf-8", "ignore").decode("utf-8"))`.
2. Given the `text` field of each jsonl record, calculate the sha1 hash of the text: `hashlib.sha1(text.encode("utf-8")).hexdigest()`.
3. Look at the two characters at the beginning of the hash:
    1. If 'ff' or 'fe', put the record in the test split.
    2. If 'fd' or 'fc', put the record in the validation split.
    3. Otherwise, put the record in the training split.

## Dataset Statistics

These stats are obtained by first loading the data into AWS Athena:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS `redpajama_v1` (
    id STRING,
    source STRING,
    version STRING,
    text STRING,
    created STRING,
    added STRING,
    metadata STRUCT<length: BIGINT>
)
PARTITIONED BY (split STRING, dataset STRING)
ROW FORMAT serde 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/redpajama/v1/documents'
TBLPROPERTIES (
  'classification'='json',
  'compressionType'='gzip'
)
```

Then by repairing the table to load all partitions:

```sql
MSCK REPAIR TABLE `redpajama_v1`
```

Finally, we run the following query:

```sql
SELECT
    split,
    dataset,
    COUNT(*) as documents,
    SUM(metadata.length) as words
FROM "redpajama_v1"
GROUP by split, dataset
ORDER BY len DESC
```

| **Split** | **Dataset**       | **Documents**   | **Whitespace-separated Words** |
|-----------|-------------------|-----------------|--------------------------------|
| train     |   common_crawl    |   475,812,014   |   621,466,665,266              |
| train     |   c4              |   364,512,631   |   131,632,788,168              |
| train     |   book            |   205,543       |   17,758,504,850               |
| train     |   github          |   28,765,890    |   17,117,495,039               |
| train     |   arxiv           |   1,556,794     |   11,280,774,175               |
| train     |   wikipedia       |   29,805,038    |   10,808,656,954               |
| train     |   stackexchange   |   29,795,923    |   9,199,379,630                |
| valid     |   common_crawl    |   232,235       |   300,813,638                  |
| valid     |   c4              |   178,519       |   64,554,914                   |
| valid     |   book            |   111           |   8,727,510                    |
| valid     |   github          |   13,682        |   8,409,018                    |
| valid     |   arxiv           |   752           |   5,492,183                    |
| valid     |   wikipedia       |   14,589        |   5,246,667                    |
| valid     |   stackexchange   |   14,599        |   4,518,021                    |
| test      |   common_crawl    |   231,770       |   301,888,240                  |
| test      |   c4              |   177,742       |   64,287,297                   |
| test      |   github          |   13,740        |   8,435,303                    |
| test      |   book            |   90            |   7,495,911                    |
| test      |   arxiv           |   760           |   5,345,683                    |
| test      |   wikipedia       |   14,544        |   5,275,691                    |
| test      |   stackexchange   |   14,564        |   4,460,034                    |

Totals:

| **Split** | **Documents** | **Whitespace-separated Words** |
|-----------|---------------|--------------------------------|
| train     | 930,453,833   | 819,264,264,082                |
| valid     | 454,487       | 397,761,951                    |
| test      | 453,210       | 397,188,159                    |


## Training a tokenizer

To train a tokenizer, we use records whose `id` (which is also the sha1 hash of the text) starts with `aa`--`af`. This should result in a dataset of about 60B words, which should be enough to train a BPE tokenizer. To extract the data, we run the following query in Athena:

```sql
UNLOAD (
    SELECT id, source, version
    FROM "redpajama_v1"
    WHERE split = 'train'
    AND regexp_like(id, '^a[01234567]')
)
TO 's3://ai2-llm/tokenizer/data/redpajama/v1/'
WITH (
    format='JSON',
    compression='GZIP'
)
```

The set above contains 18,181,314 documents and 16,020,537,754 words.

Then, we run the following command to train a tokenizer:

```bash
python -m olmo_tokenizer.hf.train \
    input_dir="s3://ai2-llm/tokenizer/data/redpajama/v1" \
    save_path="s3://ai2-llm/tokenizer/model/redpajama_v1_bpe_aa-af/tok" \
    normalization=NFC \
    min_sentence_length=128 \
    model=BPE
```

the resulting tokenizer is available at `s3://ai2-llm/tokenizer/model/redpajama_v1_bpe_aa-af/tok.json`

We also train a Unigram LM variant:

```bash
python -m olmo_tokenizer.hf.train \
    input_dir="s3://ai2-llm/tokenizer/data/redpajama/v1" \
    save_path="s3://ai2-llm/tokenizer/model/redpajama_v1_unigram_aa-af/tok" \
    normalization=NFC \
    min_sentence_length=128 \
    model=Unigram
```

which is available at `s3://ai2-llm/tokenizer/model/redpajama_v1_unigram_aa-af/tok.json`
