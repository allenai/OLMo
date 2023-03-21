UNLOAD (
    WITH s2orc_stats AS (
        SELECT
            id,
            year,
            title,
            abstract,
            fields_of_study,
            sha1,
            count,
            FILTER(paragraphs, x -> x.perplexity >= -20) as valid_paragraphs,
            (
                REGEXP_LIKE(top_frequencies[1].token, '^[A-Za-z][a-z]+$') AND (
                    (count > 500 AND (top_frequencies[1].count / count) <= 0.075) OR
                    (count <= 500 AND (top_frequencies[1].count / count) <= 0.3)
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
                                    -- extact rows to count
                                    paragraphs, x -> ROW(x.language, 1)
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
        FROM "temp_lucas"."s2orc_2023_01_03_clean_processed" as p
    ),
    filtered_corpus AS (
        SELECT
            id,
            sha1,
            year,
            title || CHR(10) || CHR(10) || abstract || CHR(10) || CHR(10) || (
                ARRAY_JOIN(TRANSFORM(valid_paragraphs, x -> x.text), CHR(10))
            ) as text
        FROM s2orc_stats
        WHERE
            language = 'en'
            AND count < 50000
            AND count > 500
            AND valid_top_word
            AND cardinality(valid_paragraphs) >= 5
            AND title IS NOT NULL
            AND abstract is not NULL
            AND year >= 1970
    )
    SELECT
        oa.id,
        oa.sha1,
        oa.text
    FROM "content_ext"."papers" as cp
    INNER JOIN filtered_corpus as oa
        ON cp.corpus_paper_id = oa.id
    -- WHERE cp.year < 2022 OR date(cp.pub_date) < date('2022-12-01')
    WHERE cp.year > 2022 OR date(cp.pub_date) >= date('2022-12-01')
)
TO 's3://ai2-s2-research-public/lucas/s2_oa_pretrain_data/v2/s2orc/valid'
WITH (
    format='JSON',
    compression='GZIP'
)
