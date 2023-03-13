CREATE EXTERNAL TABLE IF NOT EXISTS temp_lucas.all_oa_2023_01_03 (
    id INT,
    text STRING,
    lang STRING,
    cnt INT,
    freq STRING
)
PARTITIONED BY (id INT)
STORED AS PARQUET
LOCATION 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/stats'
tblproperties ("parquet.compression"="SNAPPY");
