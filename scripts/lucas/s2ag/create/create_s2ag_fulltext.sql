CREATE EXTERNAL TABLE IF NOT EXISTS temp_lucas.s2ag_fulltext_2023_01_03 (
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
    content STRUCT<
        source: STRUCT<
            pdfurls: ARRAY<STRING>,
            pdfsha: STRING,
            oainfo: STRUCT<
                license: STRING,
                openaccessurl: STRING,
                status: STRING
            >
        >,
        text: STRING,
        annotations: STRUCT<
            author: STRING,
            publisher: STRING,
            author_last_name: STRING,
            author_first_name: STRING,
            author_affiliation: STRING,
            title: STRING,
            venue: STRING,
            abstract: STRING,
            bib_ref: STRING,
            figure: STRING,
            paragraph: STRING,
            formula: STRING,
            table_ref: STRING,
            section_header: STRING,
            table: STRING,
            figure_caption: STRING,
            figure_ref: STRING,
            bib_author_first_name: STRING,
            bib_author_last_name: STRING,
            bib_entry: STRING,
            bib_title: STRING,
            bib_author: STRING,
            bib_venue: STRING
        >
    >,
    updated STRING

)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-s2ag/staging/2023-01-03/s2orc'
