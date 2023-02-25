UNLOAD
(
    USING EXTERNAL FUNCTION detect_dominant_language(text_col VARCHAR) RETURNS VARCHAR LAMBDA 'arn:aws:lambda:us-west-2:896129387501:function:textanalytics-udf'
    WITH s2ag_abstracts AS (
        SELECT
            p.corpusId,
            p.abstract,
            p.title
        FROM (
            SELECT
            id as paperId,
            corpusId,
            title,
            /* Construct ID for lookup of open-access info */
            CASE
                /* Use DOI if present */
                WHEN doi.id IS NOT NULL THEN doi.id
                /* Otherwise, search through the list of sourceIds for a non-null PubMedCentral ID */
                ELSE reduce(filter(sourceInfo.sourceIds, x -> x.source = 'PubMedCentral'),
                            null, (a, b) -> coalesce(a, b.id), x -> x)
                END            as oa_id,
            map(Array ['DOI','ArXiv','ACL','PubMedCentral','MAG'],
                Array [
                    doi.id,
                    reduce(filter(sourceInfo.sourceIds, x -> x.source = 'ArXiv'), null,
                            (a, b) -> coalesce(a, b.id), x -> x),
                    reduce(filter(sourceInfo.sourceIds, x -> x.source = 'ACL'), null,
                            (a, b) -> coalesce(a, b.id), x -> x),
                    reduce(filter(sourceInfo.sourceIds, x -> x.source = 'PubMedCentral'), null,
                            (a, b) -> coalesce(a, b.id), x -> x),
                    reduce(filter(sourceInfo.sourceIds, x -> x.source = 'MAG'), null,
                            (a, b) -> coalesce(a, b.id), x -> x)
                    ])         as externalIds,
            CASE
                /* Springer DOI prefixes */
                WHEN doi.id LIKE '10.1007/%'
                    OR doi.id LIKE '10.1013/%'
                    OR doi.id LIKE '10.1023/%'
                    OR doi.id LIKE '10.1038/%'
                    OR doi.id LIKE '10.1057/%'
                    OR doi.id LIKE '10.1065/%'
                    OR doi.id LIKE '10.1114/%'
                    OR doi.id LIKE '10.1134/%'
                    OR doi.id LIKE '10.1140/%'
                    OR doi.id LIKE '10.1186/%'
                    OR doi.id LIKE '10.1245/%'
                    OR doi.id LIKE '10.1251/%'
                    OR doi.id LIKE '10.1361/%'
                    OR doi.id LIKE '10.1365/%'
                    OR doi.id LIKE '10.1379/%'
                    OR doi.id LIKE '10.1381/%'
                    OR doi.id LIKE '10.1385/%'
                    OR doi.id LIKE '10.1617/%'
                    OR doi.id LIKE '10.1891/%'
                    OR doi.id LIKE '10.2165/%'
                    OR doi.id LIKE '10.26777/%'
                    OR doi.id LIKE '10.26778/%'
                    OR doi.id LIKE '10.3103/%'
                    OR doi.id LIKE '10.33283/%'
                    OR doi.id LIKE '10.3758/%'
                    OR doi.id LIKE '10.3858/%'
                    OR doi.id LIKE '10.4076/%'
                    OR doi.id LIKE '10.4333/%'
                    OR doi.id LIKE '10.5052/%'
                    OR doi.id LIKE '10.5819/%'
                    OR doi.id LIKE '10.7603/%'
                    THEN NULL
                    ELSE paperAbstract
                END as abstract,
            lastespressoupdate as updated
            FROM espresso.pq_paper
            WHERE partition_0 = '2023-01-03'
            ) p
            LEFT OUTER JOIN content_ext.s2oap_prod_s3 o
                ON p.oa_id = o.source_id
        WHERE abstract IS NOT NULL
    ),
    s2orc_ids AS (
        SELECT
            id
            FROM "s2orc_papers"."oa_releases"
            WHERE year=2023 AND month=01 AND day=01
    )
    SELECT
        s2ag_abstracts.corpusId as id,
        CONCAT(title, CHR(10), CHR(10), abstract) as text,
        detect_dominant_language(abstract) as lang,
        s2ag_abstracts.corpusId % 10 AS part_id
    FROM s2ag_abstracts
    -- exclude s2orc ids from dump
    LEFT OUTER JOIN s2orc_ids
        ON s2orc_ids.id = s2ag_abstracts.corpusId
    WHERE s2orc_ids.id IS NULL
)
TO 's3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2ag/'
WITH (
    format='PARQUET',
    partitioned_by = ARRAY['part_id']
)
