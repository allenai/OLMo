# Mixer/Deduper

This project contain two tools, the `mixer` and the `deduper`, both implemented in Rust.

## Mixer

Combines data from multiple sources into a unified output. Input data must be in [ai2 format](../README.md).
Merges the named attributes and applies the configured filters. Substitutes text in any configured spans.

### Configuration
See sample config files [mixer.json](tests/config/mixer.json) and [spans.json](test/config/spans.json)

**streams.name**
Prefix for output file name of each stream.

**streams.documents**
Input document files for each stream. Accepts a single wildcard `*` character.

**streams.attributes**
Merge attributes with the specified names. 
Looks for files by substituting `documents` with `attributes/<attribute_name≥` in the S3 path of each input document file.

**streams.output**
Output will be uploaded to the S3 `path`. Data will be coalesced into files no bigger than `max_size_in_bytes`.

**filter**
Optional content-based filtering. Default = keep everything. Documents are retained if they match any of the `include` patterns
(or if no `include` patterns are specified) AND if they match none of the `exclude` patterns. Pattern syntax is [jsonpath](https://support.smartbear.com/alertsite/docs/monitors/api/endpoint/jsonpath.html#filters).

**span_replacement**
A list of objects specifying spans of text to be replaced.

**span_replacement[].span**
A json-path expression for an attribute that contains an array of spans. Each span should be list of length three:  `[start, end, score]`.

**span_replacement[].min_score**
If the span score is less than this value, the span will not be replaced.

**span_replacement[].replacement**
The text that should be inserted in place of the span. Use `{}` to represent the original text.

**work_dir**
Input files are temporarily cached in the `input` directory. Output files are temporarily cached
in the `output` directory, then replaced with a zero-length file after a successful upload to S3.

**processes**
Number of processes to run in parallel. Default is number of CPUs.

## Deduper

Will create a set of attribute files, corresponding to the specified input document files.
The attributes will identify whether the entire document is a duplicate (based on some key), or
identify spans in the text that contain duplicate paragraphs.

Deduplication is done via an in-memory Bloom Filter, so there is a possibility of false positives.

Dropping any documents that are identified as duplicates, or deleting the duplicate paragraphs, can be 
done in a subsequent run of the `mixer`.

### Configuration
See sample config files [dedupe-by-url.json](tests/config/dedupe-by-url.json) and [dedupe-paragraphs.json](test/config/dedupe-paragraphs.json)

**name**

**documents**
S3 path patterns for input document files. Accepts a single wildcard `*` character.

**dedupe.name**
Used to name output attribute files. One output file will be created for each input document file,
where the key is obtained by substituting `documents` with `attributes/<name>`.

**dedupe.documents.key**
Use the json-path-specified field as the key for deduping. The value of the key must be a string.

**dedupe.documents.attribute_name**
Name of the attribute to set if the document is a duplicate. The value will be set to `true`.

**dedupe.paragraphs.attribute_name**
Name of the attribute that will contain spans of duplicate paragraphs. Paragraphs
are identified by splitting the `text` field by newline characters.

**bloom_filter.file**
Save the Bloom filter to this file after processing. If present at startup, the Bloom filter will be loaded from this file.

**size_in_bytes**
Size of the Bloom filter in bytes. Zero = set automatically based on expected number of unique documents.

**read_only**
If true, do not write to the Bloom filter. Useful for blocklisting.

**estimated_doc_count**
Estimated number of unique items that will be stored in the Bloom filter. Used to set the size of the Bloom filter.

**desired_false_positive_rate**
Desired false positive rate. Used to set the size of the Bloom filter.

## Development
Set up your environment with the Rust compiler:
```
make build-tools
```

To run tests: 
```shell
make test
```

Build the executables:
```
make
```
will create both `target/release/mixer` and `target/release/deduper`.


## Running

```
./target/release/mixer config/<your-config>.json
```

The `config/v1.json` and `config/v2.json` files were used to produce the `v1` and `v2` datasets. The code has
changed since then, and the config format is different, but you might find them useful as examples.

If running with lots of parallelism, you might need to increase the number of open files allowed:
```shell
ulimit -n 65536
```

Also, to avoid overwhelming the local credentials server, you should specify your credentials in env vars:
```shell
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```
