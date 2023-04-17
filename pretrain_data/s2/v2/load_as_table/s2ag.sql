CREATE EXTERNAL TABLE IF NOT EXISTS `temp_lucas`.`s2ag_v0_20230103` (
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
        title_language:STRING,
        abstract_language:STRING,
        title_perplexity:DOUBLE,
        abstract_perplexity:DOUBLE,
        title_count:INT,
        abstract_count:INT,
        top_frequencies:ARRAY<STRUCT<token:STRING,count:INT>>
    >
)
ROW FORMAT serde 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/s2/v0/documents/dataset=s2ag'
