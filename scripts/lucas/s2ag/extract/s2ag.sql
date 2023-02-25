UNLOAD(
    USING EXTERNAL FUNCTION detect_dominant_language(text_col VARCHAR) RETURNS VARCHAR LAMBDA 'arn:aws:lambda:us-west-2:896129387501:function:textanalytics-udf'
    WITH s2orc_ids AS (
        SELECT corpusid
        FROM "temp_lucas"."s2ag_fulltext_2023_01_03"
    )
    SELECT ta.*
    FROM (
        SELECT
            a.corpusid,
            CONCAT(t.title, CHR(10), CHR(10), a.abstract) as text,
            detect_dominant_language(a.abstract) as lang,
            a.corpusid % 10 AS part_id
        FROM "temp_lucas"."s2ag_abstacts_2023_01_03" as a
        INNER JOIN "temp_lucas"."s2ag_metadata_2023_01_03" AS t
        ON t.corpusid = a.corpusid
        WHERE t.title IS NOT NULL and a.abstract IS NOT NULL
    ) AS ta
    LEFT JOIN s2orc_ids as ts
      ON ts.corpusid = ta.corpusid
    WHERE ts.corpusid IS NULL
)
TO 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2ag/'
WITH (
    format='PARQUET',
    partitioned_by = ARRAY['part_id']
)
