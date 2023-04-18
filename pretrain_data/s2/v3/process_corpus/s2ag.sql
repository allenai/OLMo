UNLOAD (
    WITH filtered_corpus AS (
        SELECT
            id,
            source,
            added,
            created,
            metadata,
            (metadata.title || CHR(10) || CHR(10) || metadata.abstract) AS text,
            IF(
                metadata.year < 2022
                OR (
                    metadata.year = 2022 AND
                    date(from_iso8601_timestamp(created)) < date('2022-12-01')
                ),
                'train',
                'valid'
            ) AS split
        FROM (
            SELECT
                *,
                ARRAY_MAX(
                    TRANSFORM (
                        regexp_extract_all(metadata.abstract, '\b([A-Za-z]\s)([a-z]\s)*[A-Za-z]\b'),
                        x -> length(x)
                    ) || 0
                ) AS max_single_letter_sequence,
                FILTER(
                    metadata.sources,
                    x -> NOT REGEXP_LIKE(
                        x,
                        '^Unpaywall|MergedPDFExtraction|ScienceParseMerged|Anansi|ScienceParsePlus|Adhoc|ScienceParse|Crawler|MAG$'
                    )
                ) AS filtered_sources
            FROM "llm_s2ag_v0"
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
        WHERE (
            (
                CARDINALITY(filtered_sources) > 0 AND
                max_single_letter_sequence < 4
            ) AND (
                max_single_letter_sequence > 0 AND
                CARDINALITY(filtered_sources) = 0
            )
        )
    )
    SELECT
        id,
        ARRAY_AGG(source)[1] AS source,
        'v3' AS version,
        ARRAY_AGG(added)[1] AS added,
        ARRAY_AGG(created)[1] AS created,
        ARRAY_AGG(text)[1] AS text,
        ARRAY_AGG(metadata)[1] AS metadata,
        ARRAY_AGG(split)[1] AS split,
        CAST(id AS INT) % 10 AS part_id
    FROM filtered_corpus
    GROUP BY id
)
TO 's3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2ag'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['split', 'part_id']
)
