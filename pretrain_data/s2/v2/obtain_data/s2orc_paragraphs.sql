UNLOAD (
    WITH espresso_pq_metadata AS (
        SELECT
            DISTINCT pq.corpusid as id,
            pq.fieldsofstudy as fields_of_study,
            pq.id as sha1,
            to_iso8601(
                from_iso8601_timestamp(pq.earliestacquisitiondate)
            ) as added,
            to_iso8601(
                CASE
                    WHEN pq.pubdate IS null
                        THEN from_iso8601_timestamp(pq.earliestacquisitiondate)
                    ELSE date_parse(pq.pubdate, '%Y-%m-%d')
                END
            ) AS created
        FROM espresso.pq_paper AS pq
        INNER JOIN s2orc_papers.latest AS s2orc
            ON pq.corpusid = s2orc.id
    ),
    s2orc_open_access AS (
        SELECT
            id,
            metadata.publication_date.year AS year,
            metadata.title AS title,
            metadata.abstract AS abstract,
            content.grobid.contents AS full_text,
            TRANSFORM(
                CAST(
                    JSON_PARSE(content.grobid.annotations.paragraph)
                    AS ARRAY(json)
                ),
                x -> CAST(
                    ROW(
                        JSON_EXTRACT(x, '$.start'),
                        JSON_EXTRACT(x, '$.end'),
                        'paragraph'
                    ) AS ROW(bos INTEGER, eos INTEGER, type VARCHAR)
                )
            ) AS paragraph_loc,
            TRANSFORM(
                CAST(
                    JSON_PARSE(content.grobid.annotations.section_header)
                    AS ARRAY(json)
                ),
                x -> CAST(
                    ROW(
                        JSON_EXTRACT(x, '$.start'),
                        JSON_EXTRACT(x, '$.end'),
                        'section_header'
                    ) AS ROW(bos INTEGER, eos INTEGER, type VARCHAR)
                )
            ) AS section_header_loc
        FROM "s2orc_papers"."oa_releases"
        WHERE
            year=2023 AND
            month=01 AND
            day=01 AND
            content.grobid.contents is not null
    ),
    prepared_locs AS (
        SELECT
            id,
            year,
            title,
            abstract,
            full_text,
            ARRAY_SORT(
                ARRAY_DISTINCT(paragraph_loc || section_header_loc)
            ) AS all_paralocs
        FROM s2orc_open_access
    ),
    extracted_paragraphs AS (
        SELECT
            id,
            year,
            title,
            abstract,
            TRANSFORM(
                all_paralocs,
                x -> CAST(
                     ROW(
                        SUBSTR(full_text, x.bos, x.eos - x.bos + 1),
                        x.type
                    ) AS ROW(text VARCHAR, type VARCHAR)
                )
            ) AS all_paragraphs
        FROM prepared_locs
    )
    SELECT
        pt.*,
        ep.fields_of_study,
        ep.sha1,
        ep.added,
        ep.created,
        -- make 100 partitions for smaller output files
        pt.id % 100 as part_id
    FROM extracted_paragraphs AS pt
    INNER JOIN espresso_pq_metadata AS ep
        ON pt.id = ep.id
)
TO 's3://ai2-llm/pretraining-data/sources/s2/raw/2023_01_03/s2orc/'
WITH (
    format='PARQUET',
    partitioned_by = ARRAY['part_id']
)
