UNLOAD (
    WITH s2orc_stats AS (
        SELECT
            id,
            source,
            added,
            created,
            metadata,
            FILTER(
                metadata.paragraphs,
                x -> x.perplexity >= -20
            ) as valid_paragraphs,
            (
                REGEXP_LIKE(
                    metadata.top_frequencies[1].token,
                    '^[A-Za-z][a-z]+$'
                ) AND (
                    (
                        metadata.count > 500 AND
                        (
                            metadata.top_frequencies[1].count / metadata.count
                        ) <= 0.075
                    ) OR (
                        metadata.count <= 500 AND
                        (
                            metadata.top_frequencies[1].count / metadata.count
                        ) <= 0.3
                    )
                )
            ) AS valid_top_word,
            ARRAY_SORT(
                TRANSFORM(
                    MAP_ENTRIES(
                        TRANSFORM_VALUES(
                            -- from table to map
                            MULTIMAP_FROM_ENTRIES(
                                -- from list to table
                                TRANSFORM(
                                    -- extract rows to count
                                    metadata.paragraphs,
                                    x -> ROW(x.language, 1)
                                )
                            ),
                            -- merge counts
                            (k, v) -> REDUCE(v, 0, (s, x) -> s + x, s -> s)
                        )
                    ),
                    x -> CAST(x AS ROW(lang varchar, cnt int))
                ),
                (x, y) -> IF(x.cnt < y.cnt, 1, IF(x.cnt = y.cnt, 0, -1))
            )[1].lang AS language
        FROM "temp_lucas"."llm_s2orc_v0"
    ),
    filtered_corpus AS (
        SELECT
            id,
            source,
            added,
            created,
            metadata,
            cast(id AS INT) as corpusid,
            (
                metadata.title || CHR(10) || CHR(10) ||
                metadata.abstract || CHR(10) || CHR(10) ||
                ARRAY_JOIN(TRANSFORM(valid_paragraphs, x -> x.text), CHR(10))
            ) as text,
            IF(
                metadata.year < 2022
                OR (
                    metadata.year = 2022 AND
                    date(from_iso8601_timestamp(created)) < date('2022-12-01')
                ),
                'train',
                'valid'
            ) AS split
        FROM s2orc_stats
        WHERE
            language = 'en'
            AND metadata.count < 50000
            AND metadata.count > 500
            AND valid_top_word
            AND cardinality(valid_paragraphs) >= 5
            AND metadata.title IS NOT NULL
            AND metadata.abstract is not NULL
            AND metadata.year >= 1970
    ),
    filtered_corpus_with_fos AS (
        SELECT
            cr.id,
            cr.source,
            cr.added,
            cr.created,
            cr.text,
            cr.split,
            CAST(
                ROW(
                    cr.metadata.year,
                    cr.metadata.title,
                    cr.metadata.abstract,
                    cr.metadata.sha1,
                    cr.metadata.paragraphs,
                    cr.metadata.count,
                    cr.metadata.top_frequencies,
                    COALESCE(pq.s2FieldsOfStudy, ARRAY[]),
                    COALESCE(pq.fieldsOfStudy, ARRAY[])
                )
                AS
                ROW(
                    year BIGINT,
                    title VARCHAR,
                    abstract VARCHAR,
                    sha1 VARCHAR,
                    paragraphs ARRAY<ROW(language VARCHAR, perplexity DOUBLE, text VARCHAR)>,
                    count BIGINT,
                    top_frequencies ARRAY<ROW(token VARCHAR, count BIGINT)>,
                    s2FieldsOfStudy ARRAY<VARCHAR>,
                    extFieldsOfStudy ARRAY<VARCHAR>
                )
            ) AS metadata
        from espresso.pq_paper  as pq
        INNER JOIN filtered_corpus as cr
            ON pq.corpusid = cr.corpusid
    )
    SELECT
        id,
        ARRAY_AGG(source)[1] AS source,
        'v3-fos' AS version,
        ARRAY_AGG(added)[1] AS added,
        ARRAY_AGG(created)[1] AS created,
        ARRAY_AGG(text)[1] AS text,
        ARRAY_AGG(metadata)[1] AS metadata,
        ARRAY_AGG(split)[1] AS split,
        CAST(id AS INT) % 10 AS part_id
    FROM filtered_corpus
    GROUP BY id
)
TO 's3://ai2-llm/pretraining-data/sources/s2/v3-fos/documents/dataset=s2orc'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['split', 'part_id']
)
