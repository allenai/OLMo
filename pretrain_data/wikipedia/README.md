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
