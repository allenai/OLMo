# Data For Tokenizer



##

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

Data resulting from this query is located at `s3://ai2-llm/tokenizer/data/v1`. The
