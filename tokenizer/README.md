# Data For Tokenizer


## 2023-04-04

### Source Stats

Words are identified by the following regex: `\w+|[^\w\s]+` (borrowed from Huggingface's tokenizers library).

| **Dataset**        | **Docs**         | **Words**           |
|:------------------:|:----------------:|:-------------------:|
| S2AG               |  59,382,301      |  12,803,090,966     |
| S2ORC (Abstracts)  |   8,242,162      |   2,110,166,352     |
| Wikipedia (EN)     |   4,199,703      |   2,978,530,335     |
| Wikipedia (non EN) |  20,022,061      |  10,848,243,609     |
| C4                 | 364,156,258      | 156,647,005,716     |



### Data

We decided to train a tokenizer on all of Wikipedia, a subset of C4, and a subset of S2 abstracts.
In detail, our mixing strategy is as follows:

- 100% of wikipedia (~3B english tokens, ~8B non-english tokens)
- 34% of C4 (~53B english tokens)
- 64% of S2 abstracts (~8B english tokens)

The split above give us roughly

As Athena Query:

```sql
UNLOAD (
    WITH wiki AS (
        SELECT text
        -- from here we sample ~3M english and ~8M non english
        FROM "llm_wikipedia_v0"
        WHERE (
            -- low quality languages (auto generated or known vandalism)
            lang != 'ceb' and lang != 'sv' and lang != 'sco'
            -- at least 100 words in each article
            and metadata.length >= 100
        )
    ),
    s2 AS (
        SELECT ARRAY_JOIN(
            -- only keep the first two blocks of text separated by \n\n
            -- which are title and abstract
            SLICE(SPLIT(text, CHR(10) || CHR(10), 3), 1, 2), CHR(10)
        ) as text
        FROM "temp_lucas"."llm_s2_v2"
        -- 0.64 sample rate gets us ~15% of S2 data
        WHERE split = 'train' AND RAND() < 0.64
    ),
    c4 AS (
        SELECT text
        FROM "temp_lucas"."llm_c4_v0"
        WHERE RAND() < 0.34
    )
    SELECT
        *,
        CAST(ROUND(RANDOM(10)) AS INT) AS part_id
    FROM (
        SELECT * FROM wiki
        UNION
        SELECT * FROM s2
        UNION
        SELECT * FROM c4
    )
)
TO 's3://ai2-llm/tokenizer/data/v1'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by=ARRAY['part_id']
)
```

### Model

A model can be trained with the following command:

```bash
python -m olmo_tokenizer.train_v1
```

Resulting model is uploaded at `s3://ai2-llm/tokenizer/model/v1.json`. A model trained on 10% of the data is available at `s3://ai2-llm/tokenizer/model/v1_small.json`. Another trained on 1% of the data is at `s3://ai2-llm/tokenizer/model/v1_tiny.json`.


## 2023-04-06

We noticed issue.


### Data Refresh (V2)


```sql
UNLOAD (
    WITH wiki AS (
        SELECT text, if(lang = 'en', 'wiki-en', 'wiki') as source
        -- from here we sample ~3M english and ~8M non english
        FROM "llm_wikipedia_v0"
        WHERE (
            -- low quality languages (auto generated or known vandalism)
            lang != 'ceb' and lang != 'sv' and lang != 'sco'
            -- at least 100 words in each article
            and metadata.length >= 100
        )
    ),
    s2 AS (
        SELECT ARRAY_JOIN(
            -- only keep the first two blocks of text separated by \n\n
            -- which are title and abstract
            SLICE(SPLIT(text, CHR(10) || CHR(10), 3), 1, 2), CHR(10)
        ) as text, 's2' as source
        FROM "temp_lucas"."llm_s2_v2"
        -- 0.64 sample rate gets us ~15% of S2 data
        WHERE split = 'train' AND RAND() < 0.73
    ),
    c4 AS (
        SELECT text, 'c4' as source
        FROM "temp_lucas"."llm_c4_v0"
        WHERE RAND() < 0.38
    )
    SELECT
        *,
        CAST(ROUND(RANDOM(10)) AS INT) AS part_id
    FROM (
        SELECT * FROM wiki
        UNION
        SELECT * FROM s2
        UNION
        SELECT * FROM c4
    )
)
TO 's3://ai2-llm/tokenizer/data/v2'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by=ARRAY['part_id']
)
```
