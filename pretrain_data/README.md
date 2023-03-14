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

#### `ai2` format

This is the unified format we will use across all the sources. All data that lives in `v0`, `v1`, ... will be in this format.

`v0` looks like:

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


And later `v1`, `v2`, ... will look the same but with additional keys:

```
"lang": {"en": 0.9, "fr": 0.2, "de": 0.1 }
"toxicity": 0.7,
```


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


