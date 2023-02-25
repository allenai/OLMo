UNLOAD(
    USING EXTERNAL FUNCTION detect_dominant_language(text_col VARCHAR) RETURNS VARCHAR LAMBDA 'arn:aws:lambda:us-west-2:896129387501:function:textanalytics-udf'
    SELECT
        corpusid,
        content.text as text,
        detect_dominant_language(substr(content.text, 1, 1000)) as lang,
        corpusid % 10 AS part_id
    FROM "temp_lucas"."s2ag_fulltext_2023_01_03"
    WHERE content.text is not NULL
)
TO 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc/'
WITH (
    format='PARQUET',
    partitioned_by = ARRAY['part_id']
)
