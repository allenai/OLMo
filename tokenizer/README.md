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

We noticed issue with the tokenizer, which caused many words to be split into too many tokens. For example, consider the following statistics on LAMBADA:

```
lambada_openai/ai2_v1
avg: 100.20 std: 16.80
reversible tokenization: 97.90%
Most common tokens:
' ': 7.78%
'.': 4.26%
',': 3.63%
'he': 2.71%
' t': 2.70%

lambada_openai/ai2_v1_en
avg: 101.49 std: 17.08
reversible tokenization: 97.90%
Most common tokens:
' ': 10.48%
't': 6.06%
'.': 4.21%
',': 3.58%
'o': 2.82%

lambada_openai/ai2_v1_en_nos2
avg: 93.99 std: 15.19
reversible tokenization: 97.90%
Most common tokens:
' ': 6.71%
'.': 4.54%
',': 3.87%
'e': 3.46%
'th': 3.16%

lambada_openai/gpt2
avg: 83.49 std: 13.07
reversible tokenization: 97.55%
Most common tokens:
'.': 4.33%
',': 3.99%
'\n': 3.48%
' the': 3.23%
'�': 2.46%

lambada_openai/bloom
avg: 79.27 std: 12.54
reversible tokenization: 97.55%
Most common tokens:
',': 4.58%
'.': 3.57%
' the': 3.40%
' to': 2.20%
' and': 1.69%
```

Note the large number of tokens produced until we removed S2 data. Looking at the following example, it is even more apparent what is going on (see how `it` and `is` get split into 3 tokens each):

```
Original
In my palm is a clear stone, and inside it is a small ivory statuette.

ai2_v1 (PreTrainedTokenizerFast)
In Ġmy Ġpalm Ġ i s Ġ a Ġclear Ġstone , Ġ an d Ġinside Ġ i t Ġ i s Ġ a Ġsmall Ġivory Ġsta tu ette .

ai2_v1_en (PreTrainedTokenizerFast)
In Ġmy Ġpalm Ġ i s Ġ a Ġclear Ġstone , Ġ an d Ġinside Ġ i t Ġ i s Ġ a Ġsmall Ġivory Ġstatue tte .

ai2_v1_en_no_s2 (PreTrainedTokenizerFast)
In Ġmy Ġpalm Ġis Ġa Ġclear Ġstone , Ġand Ġinside Ġit Ġis Ġa Ġsmall Ġivory Ġstat u ette .

gpt2 (GPT2TokenizerFast)
In Ġmy Ġpalm Ġis Ġa Ġclear Ġstone , Ġand Ġinside Ġit Ġis Ġa Ġsmall Ġivory Ġstat u ette .

bloom (PreTrainedTokenizerFast)
In Ġmy Ġpalm Ġis Ġa Ġclear Ġstone , Ġand Ġinside Ġit Ġis Ġa Ġsmall Ġiv ory Ġstatu ette .
```

We found the issue to be by s2 abstracts containing extraneous spaces between letter in contiguous words (this is likely due to OCR errors). I grepped C4, S2, and Wikipedia corpora for `\bi\ss\b`  and counted number of docs matching. Here's what I found: while c4 has the lowest: of the 365M docs, it only occurs ~12k or 0.003%, s2 has 0.021% or 14k docs (with the full-text papers subset being only 0.009% or 779 docs).

To fix the issue, we crated version 3 of the S2 data, which is described [in this document](../pretrain_data/s2/v3/README.md). Based on it, we derived the following data mix:

| **Dataset**        | **Docs**         | **Words**           |
|:------------------:|:----------------:|:-------------------:|
| S2AG               |  30,569,017      |   6,721,301,098     |
| S2ORC (Abstracts)  |   8,242,162      |   2,110,166,352     |
| Wikipedia (EN)     |   4,284,837      |   3,005,337,486     |
| Wikipedia (non EN) |  19,936,927      |  10,821,436,458     |
| C4                 | 364,156,258      | 156,647,005,716     |

We sample 38% of C4 to obtain a ~82B tokens to train a tokenizer. See the script below for sampling strategy.


### Data Refresh (V2)


```sql
UNLOAD (
    WITH wiki AS (
        SELECT text, if(lang = 'en' OR lang = 'simple', 'wiki-en', 'wiki') as source
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
        ) as text,
        dataset as source
        FROM "temp_lucas"."llm_s2_v3"
        -- 0.64 sample rate gets us ~15% of S2 data
        WHERE split = 'train'
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
