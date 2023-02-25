UNLOAD(
    USING EXTERNAL FUNCTION detect_dominant_language(text_col VARCHAR) RETURNS VARCHAR LAMBDA 'arn:aws:lambda:us-west-2:896129387501:function:textanalytics-udf'
    SELECT
        id,
        content.grobid.contents as text,
        detect_dominant_language(substr(content.grobid.contents, 1, 1000)) as lang,
        id % 10 as part_id
    FROM "s2orc_papers"."oa_releases"
    WHERE year=2023 AND month=01 AND day=01 AND content.grobid.contents is not null
)
TO 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc/'
WITH (
    format='PARQUET',
    partitioned_by = ARRAY['part_id']
)
