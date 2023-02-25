CREATE EXTERNAL TABLE IF NOT EXISTS temp_lucas.s2ag_metadata_2023_01_03 (
    corpusid BIGINT,
    externalids STRUCT<
        ACL: STRING,
        DBLP: STRING,
        ArXiv: STRING,
        MAG: STRING,
        CorpusId: STRING,
        PubMed: STRING,
        DOI: STRING,
        PubMedCentral: STRING
    >,
    url STRING,
    title STRING,
    authors ARRAY<STRUCT<
        authorId: STRING,
        name: STRING
    >>,
    venue STRING,
    publicationvenueid STRING,
    year INT,
    referencecount INT,
    citationcount INT,
    influentialcitationcount INT,
    isopenaccess BOOLEAN,
    s2fieldsofstudy ARRAY<STRING>,
    publicationtypes ARRAY<STRING>,
    publicationdate STRING,
    journal STRUCT<
        name: STRING,
        pages: STRING,
        volume: STRING
    >,
    updated STRING
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-s2ag/staging/2023-01-03/papers'
