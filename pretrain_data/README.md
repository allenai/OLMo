# pretrain-data

working place for pretraining data curation. please contact:
- common crawl: rodneyk@allenai.org
- semantic scholar data: lucas@allenai.org
- stack (code): akshitab@allenai.org
- reddit: dustins@allenai.org
- wikipedia:
- books:
- 

### organizational structure

All our data lives at `s3://ai2-llm/pretraining-data/` and looks like:

```
sources/
|-- common-crawl/
    |-- raw/
    |-- v0/
        |-- 2019-09/
            |-- 0933_uk_all.jsonl.gz        (1GB)
            |-- 0933_vi_all.jsonl.gz        (1GB)
            |-- 0106_uk_all.jsonl.gz        (1GB)
            |-- 0106_vi_all.jsonl.gz        (1GB)
        |-- 2019-08/
    |-- v1/
|-- s2/
    |-- raw/
    |-- v0/
        |-- 6f5c/
            |-- 7b3dbd9826c5e8d489fff2955ced.jsonl.gz      (1GB)
            |-- a98fe533696c63b5a39c4c45c3c7.jsonl.gz      (1GB)
        |-- 766f/
    |-- v1/
|-- stack/
|-- reddit/
|-- wiki/
```

#### `raw`

Each source organizes its `raw` data however it likes. The key is to preserve whatever the data looks like at the earliest stage in case we need to ever regenerate our conversion from `raw` format to `ai2` format. 

To keep things simple, we recommend every owner of a source to maintain something akin to:
```python
|-- common_crawl/
    |-- raw_to_ai2_format.py
|-- s2/
    |-- raw_to_ai2_format.py
|-- stack/
    |-- raw_to_ai2_format.py
|-- ...
```


#### `ai2` format

This is the unified format we will use across all the sources. All data that lives in `v0`, `v1`, ... will be in this format.

Each row in one of the `v0` JSONLs looks like:

```
{
    "source": "...",         # 'wiki', 'cc', 's2', ...
    "id": "...",             # source-specific
    "text": "foo",
    "added": "...",          # timestamp acquired this data
    "timestamp": "..."       # timestamp of the orig document (best-guess)
    "metadata": {...}        # source-specific metadata
}
```


And later rows in `v1`, `v2`, ... will look the same but with additional keys:

```
"lang": {"en": 0.9, "fr": 0.2, "de": 0.1 }
"toxicity": 0.7,
```

##### versions

To go from `v0` to `v1` to `v2`, we will employ tools:

* `Mapper` takes a single `/source/*/*.jsonl.gz` and returns a `/source/*/*.jsonl.gz` with the same number of rows, in the same order. Bug the `Mapper` adds new keys/fields to each JSON object. This will be used to do things like `LanguageId` or `ToxicityDetect` per document.  
* `Filterer` also takes a single `/source/*/*.jsonl.gz` and returns a `/source/*/*.jsonl.gz`, but it returns fewer rows than before. 
* `Mixer` takes multiple `/source/*/*.jsonl.gz` files and a `config` file and returns multiple `/source/*/*.jsonl.gz` files. This is typically used to mix input data dumps from different sources (e.g. Wiki + S2 + CommonCrawl + Stack) and returns a single mixture of them according to some specification. (e.g. % of each, sorted in a certain way). 
* `Deduper` TBD. `ianm@allenai.org` and `rodneyk@allenai.org` are working something out.

But the idea is that the input & output of these tools always preserves the JSON format & at best removes/adds rows per `jsonl.gz` dump and/or adds more keys to the JSON lines. 

More details can be found in this [Proposal Doc](https://docs.google.com/document/d/18T5_v3QeWPiiuSsUi09_6-ZxW_i47cBatblABb9IZ0M/edit?usp=sharing).

##### `id`

The `id` field is very important as we will need the ability to store a `blocklist` of documents (e.g. avoid due to LLM-Eval, takedown requests, manual inspection). 

It is important that one can uniquely identify a single document in all data versions from a given `id`. That means we can pinpoint that document ABC in `raw` is the same as document ABC in `v0`, `v1`, ...

Otherwise, the `id` only needs to be consistent/unique within a `source`. So `id=123` is acceptable because `(c4, 123)` and `(github, 123)` would uniquely identify this document still.

  

##### `metadata`

The `metadata` field will be a free-for-all field that contains any source-specific information. This could be things like code license for the Stack, or paper identifiers for S2 data.






### open questions

1. Where should "shared" utilities live? What are they?
2. Shared dependencies or treat each source as own project?
3. Comfortable pushing directly, or pull requests w/ reviews? Who approving?
4. Keeping developer logs. GitHub likely better than Google Docs. Maybe define a minimal convention/cadence for this? Can be rough. 


