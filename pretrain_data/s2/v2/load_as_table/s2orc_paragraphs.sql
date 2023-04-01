
CREATE EXTERNAL TABLE IF NOT EXISTS `temp_lucas`.`s2orc_v0_20230103` (
    id STRING,
    source STRING,
    text STRING,
    version STRING,
    added STRING,
    created STRING,
    metadata STRUCT<
        year:INT,
        title:STRING,
        abstract:STRING,
        sha1:STRING,
        fields_of_study:ARRAY<STRING>,
        paragraphs:ARRAY<STRUCT<language:STRING,perplexity:DOUBLE,text:STRING>>,
        count:INT,
        top_frequencies:ARRAY<STRUCT<token:STRING,count:INT>>
    >
)
ROW FORMAT serde 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/s2/v0/documents/dataset=s2orc'
