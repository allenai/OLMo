CREATE EXTERNAL TABLE IF NOT EXISTS temp_lucas.s2ag_abstacts_2023_01_03 (
    corpusid BIGINT,
    openaccessinfo STRUCT<
        externalids: STRUCT<
            MAG: STRING,
            ACL: STRING,
            DOI: STRING,
            PubMedCentral: STRING,
            ArXiv: STRING
        >,
        license: STRING,
        url: STRING,
        status: STRING
    >,
    abstract STRING,
    updated STRING
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-s2ag/staging/2023-01-03/abstracts'
