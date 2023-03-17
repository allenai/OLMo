CREATE EXTERNAL TABLE IF NOT EXISTS `temp_lucas`.`s2orc_2023_01_03_clean_processed` (
    id INT,
    year INT,
    title STRING,
    abstract STRING,
    filtered_paragraphs ARRAY<STRING>,
    fields_of_study ARRAY<STRING>,
    sha1 VARCHAR(40),
    language_paragraphs ARRAY<STRING>,
    words_paragraphs ARRAY<ARRAY<STRING>>,
    perplexity_paragraphs ARRAY<FLOAT>,
    count INT,
    top_frequencies STRING
)
STORED AS PARQUET
LOCATION 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc_clean_processed/'
tblproperties ("parquet.compression"="SNAPPY");
