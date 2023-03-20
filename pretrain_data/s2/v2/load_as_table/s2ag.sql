CREATE EXTERNAL TABLE IF NOT EXISTS `temp_lucas`.`s2ag_2023_01_03_clean_processed` (
    id INT,
    year INT,
    title STRING,
    abstract STRING,
    sha1 STRING,
    title_language STRING,
    abstract_language STRING,
    title_perplexity DOUBLE,
    abstract_perplexity DOUBLE,
    title_count INT,
    abstract_count INT,
    top_frequencies ARRAY<STRUCT<token:STRING,count:INT>>
)
STORED AS PARQUET
LOCATION 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2ag_clean_processed/'
tblproperties ("parquet.compression"="SNAPPY");
