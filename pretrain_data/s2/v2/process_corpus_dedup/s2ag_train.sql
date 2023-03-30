UNLOAD (
    WITH filtered_corpus AS (
        SELECT
            id,
            source,
            version,
            added,
            created,
            metadata,
            (metadata.title || CHR(10) || CHR(10) || metadata.abstract) AS text
        FROM "temp_lucas"."s2ag_v0_20230103"
        WHERE
            (metadata.title_language = 'en' OR metadata.title_perplexity > -20)
            AND metadata.abstract_language = 'en'
            AND metadata.abstract_perplexity > -20
            AND metadata.title_count >= 3
            AND metadata.abstract_count >= 50
            AND metadata.abstract_count <= 1000
            AND metadata.year >= 1970
            AND (
                REGEXP_LIKE(
                    metadata.top_frequencies[1].token,
                    '^[A-Za-z][A-Za-z]+$'
                )
                OR (
                    metadata.top_frequencies[1].token = 'a'
                    AND REGEXP_LIKE(
                        metadata.top_frequencies[2].token,
                        '^[A-Za-z][A-Za-z]+$'
                    )
                )
            )
        )
    SELECT
        id,
        ARRAY_AGG(source)[1] AS source,
        ARRAY_AGG(version)[1] AS version,
        ARRAY_AGG(added)[1] AS added,
        ARRAY_AGG(created)[1] AS created,
        ARRAY_AGG(text)[1] AS text,
        ARRAY_AGG(metadata)[1] AS metadata,
        CAST(id AS INT) % 10 AS part_id
    FROM (
        SELECT *
        FROM filtered_corpus
        WHERE metadata.year < 2022
            OR date(from_iso8601_timestamp(created)) < date('2022-12-01')
    )
    GROUP BY id
)
TO 's3://ai2-llm/pretraining-data/sources/s2/v2_dedup/documents/dataset=s2ag/split=train'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['part_id']
)
