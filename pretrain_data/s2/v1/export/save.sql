UNLOAD (
    SELECT
        id,
        text
    FROM (
        SELECT
            id,
            text,
            cnt,
            max_freq.word as top_word,
            max_freq.cnt as top_cnt,
            CAST(max_freq.cnt AS REAL) / CAST(cnt AS REAL) as top_frac
        FROM (
            SELECT
                id,
                text,
                cnt,
                CAST(json_array_get(freq, 0) AS ROW(word VARCHAR, cnt INT)) as max_freq
            FROM "temp_lucas"."all_oa_2023_01_03"
            WHERE lang = 'en' AND cnt < 50000 AND cnt > 50
        )
    )
    WHERE
        REGEXP_LIKE(top_word, '^[A-Za-z][a-z]+$') AND (
            (cnt > 500 AND top_frac <= 0.075) OR
            (cnt <= 500 AND top_frac <= 0.3)
        )
)
TO 's3://ai2-s2-research-public/lucas/s2orc_oa_2022_01_03'
WITH (
    format='JSON',
    compression='GZIP'
)
