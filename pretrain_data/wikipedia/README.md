# Wikipedia

Curator: lucas@allenai.org

## Downloading Wikipedia Dumps

XML dumps are available at [dumps.wikimedia.org](https://dumps.wikimedia.org/).
As of 2023-03-25, [this page](https://en.wikipedia.org/wiki/List_of_Wikipedias#Number_of_Wikipedias_by_language_families_and_groups) lists the following Wikipedias as having more than 500,000 articles:

| **Language**       | **Language (local)** | **Wiki** | **Articles**   |
|--------------------|----------------------|:--------:|:--------------:|
| English            | English              | en       |  6,634,914     |
| ~~Cebuano~~        | ~~Cebuano~~          | ~~ceb~~  |  ~~6,123,642~~ |
| German             | Deutsch              | de       |  2,785,380     |
| ~~Swedish~~        | ~~svenska~~          | ~~sv~~   |  ~~2,560,024~~ |
| French             | français             | fr       |  2,508,170     |
| Dutch              | Nederlands           | nl       |  2,119,252     |
| Russian            | русский              | ru       |  1,903,561     |
| Spanish            | español              | es       |  1,849,264     |
| Italian            | italiano             | it       |  1,803,910     |
| Egyptian Arabic    | مصرى                 | arz      |  1,617,138     |
| Polish             | polski               | pl       |  1,561,589     |
| Japanese           | 日本語                | ja       |  1,367,773     |
| Chinese            | 中文                  | zh       |  1,342,848     |
| Vietnamese         | Tiếng Việt           | vi       |  1,282,016     |
| Waray              | Winaray              | war      |  1,266,084     |
| Ukrainian          | українська           | uk       |  1,252,455     |
| Arabic             | العربية              | ar       |  1,203,486     |
| Portuguese         | português            | pt       |  1,103,131     |
| Persian            | فارسی                | fa       |  956,293       |
| Catalan            | català               | ca       |  724,074       |
| Serbian            | српски / srpski      | sr       |  669,364       |
| Indonesian         | Bahasa Indonesia     | id       |  641,887       |
| Korean             | 한국어                 | ko       |  629,502       |
| Norwegian (Bokmål) | norsk                | no       |  608,140       |
| Chechen            | нохчийн              | ce       |  561,502       |
| Finnish            | suomi                | fi       |  549,610       |
| Hungarian          | magyar               | hu       |  522,179       |
| Czech              | čeština              | cs       |  521,418       |
| Turkish            | Türkçe               | tr       |  515,542       |
|                    |                      |          |                |
| **Total**          |                      |          | **47,184,148** |
| **Total** -ceb -sv |                      |          | **38,500,482** |

We skip Cebuano (ceb) and Swedish (sv) because they contain a [large number of machine-generated articles](https://blog.datawrapper.de/wikipedia-articles-written-by-a-bot/) of dobious quality.
The generated articles are created by [Lsjbot](<https://en.wikipedia.org/wiki/Lsjbot>).

## Downloading Wikipedia Articles

We use dumps from 2020-03-20. The have the following format:

```plain-text
https://dumps.wikimedia.org/{lang_code}wiki/20230320/{lang_code}wiki-20230320-pages-articles-multistream.xml.bz2
```

where `{lang_code}` is the language code from the table above.

In order to download the dumps, first install the `pretrain_data_wikipedia` package:

```bash
pip install ./pretrain_data/wikipedia
```

Then, run the following command:

```bash
python -m pretrain_data_wikipedia.download \
    local_dst=/net/nfs2.s2-research/lucas/wikipedia \
    debug=false \
    parallel=3
```

It doesn't seem to be possible to download more than 3 in parallel. Speed seems to be limited to ~5MiB/s per connection.


## Raw Wikipedia Data

The raw Wikipedia data is available at `s3://ai2-llm/pretrain_data/wikipedia/raw/`.
License for the text data is [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
More information about the license can be found [here](https://dumps.wikimedia.org/legal.html).

## Processing wikipedia Data

Follow these steps to process the raw Wikipedia data into a format suitable for training a language model.

1. Install the `pretrain_data_wikipedia` package:

    ```bash
    pip install ./pretrain_data/wikipedia
    ```

2. Run the following command to download the raw Wikipedia data:

    ```bash
    python -m pretrain_data_wikipedia.download \
        local_dst=/net/nfs2.s2-research/lucas/wikipedia \
        debug=false \
        parallel=3
    ```

3. Use WikiExtractor to extract the text from the Wikipedia dumps:

    ```bash
    bash extract_all.sh \
        -i /net/nfs2.s2-research/lucas/wikipedia \
        -o /net/nfs2.s2-research/lucas/wikipedia-processed
    ```

4. Compress the data:

    ```bash
    python -m pretrain_data_wikipedia.compress \
        local_src=/net/nfs2.s2-research/lucas/wikipedia-processed \
        local_dst=/net/nfs2.s2-research/lucas/wikipedia-processed-compressed \
        parallel=4
    ```


## V0 Statistics

Data is available at `s3://ai2-llm/pretrain_data/wikipedia/v0/`.

Load it into Athena with:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS `llm_wikipedia_v0` (
    id STRING,
    revid STRING,
    url STRING,
    title STRING,
    text STRING
)
PARTITIONED BY (lang STRING)
ROW FORMAT serde 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/wikipedia'
```

and then run to scan partitions:


```sql
MSCK REPAIR TABLE `llm_wikipedia_v0`
```

After loading the data into Athena, we use the following query to
obtain a count of the number of articles and tokens per language

```sql
SELECT
    lang,
    COUNT(cnt) AS docs_count,
    SUM(cnt) AS tokens_count
FROM (
    SELECT
        lang,
        CARDINALITY(
            filter(
                REGEXP_SPLIT(text, '\s+'),
                x -> LENGTH(TRIM(x)) > 0
            )
        ) AS cnt
    FROM "llm_wikipedia_v0"
    WHERE text != ''
)
GROUP BY lang
ORDER BY tokens_count DESC
```

Here are the numbers of documents and whitespace-separated tokens
for each language.


|  **lang**  |  **Documents**   |  **Whitespace Tokens**  |
|:----------:|:----------------:|:-----------------------:|
|  en        |  6,597,413       |  2,526,060,496          |
|  de        |  2,762,076       |  928,634,657            |
|  fr        |  2,435,743       |  835,619,681            |
|  es        |  1,779,272       |  712,188,125            |
|  ru        |  1,847,443       |  556,025,623            |
|  it        |  1,617,797       |  526,008,714            |
|  pt        |  1,072,432       |  309,469,768            |
|  nl        |  2,032,988       |  286,424,349            |
|  pl        |  1,490,808       |  259,003,769            |
|  uk        |  1,182,295       |  258,671,123            |
|  ca        |  696,864         |  231,510,644            |
|  ar        |  1,192,969       |  194,280,039            |
|  vi        |  1,279,525       |  182,010,536            |
|  cs        |  515,441         |  144,192,588            |
|  hu        |  491,174         |  123,500,293            |
|  fa        |  952,157         |  116,977,514            |
|  no        |  599,353         |  105,395,622            |
|  id        |  628,169         |  102,552,970            |
|  sr        |  697,921         |  100,889,886            |
|  fi        |  542,515         |  93,396,202             |
|  tr        |  504,341         |  77,237,014             |
|  ko        |  621,483         |  76,929,773             |
|  arz       |  1,613,920       |  47,167,388             |
|  war       |  1,266,646       |  41,849,042             |
|  simple    |  225,835         |  27,147,943             |
|  ce        |  558,005         |  24,132,558             |
|  ja        |  1,358,738       |  23,228,263             |
|  zh        |  1,331,870       |  12,407,485             |
