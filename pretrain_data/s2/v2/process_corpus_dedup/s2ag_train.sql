UNLOAD (
    WITH filtered_corpus AS (
        SELECT
            id,
            sha1,
            title || CHR(10) || CHR(10) || abstract AS text
        FROM "temp_lucas"."s2ag_2023_01_03_clean_processed"
        WHERE
            (title_language = 'en' OR title_perplexity > -20)
            AND abstract_language = 'en'
            AND abstract_perplexity > -20
            AND title_count >= 3
            AND abstract_count >= 50
            AND abstract_count <= 1000
            AND year >= 1970
            AND (
                REGEXP_LIKE(top_frequencies[1].token, '^[A-Za-z][A-Za-z]+$')
                OR (
                    top_frequencies[1].token = 'a'
                    AND REGEXP_LIKE(top_frequencies[2].token, '^[A-Za-z][A-Za-z]+$')
                )
            )
        )
    SELECT
        id,
        sha1,
        ARBITRARY(text) as text
    FROM (
        SELECT
            oa.id,
            oa.sha1,
            oa.text
        FROM "content_ext"."papers" as cp
        INNER JOIN filtered_corpus as oa
            ON cp.corpus_paper_id = oa.id
        WHERE cp.year < 2022 OR date(cp.pub_date) < date('2022-12-01')
    )
    GROUP BY id, sha1
)
TO 's3://ai2-llm/pretraining-data/sources/s2/v2_dedup/dataset=s2ag/split=train'
WITH (
    format='JSON',
    compression='GZIP'
)
