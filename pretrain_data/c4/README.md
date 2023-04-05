# C4

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS `llm_c4_v0` (
    id STRING,
    text STRING,
    added STRING,
    created STRING,
    metadata STRUCT<
        lang: STRING,
        url: STRING,
        length: BIGINT
    >
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/c4/v0/documents'
TBLPROPERTIES (
  'classification'='json',
  'compressionType'='gzip'
)
```
