## Dataset Summary

We ran the CCNet pipeline over 25 dumps from 2020-05 to 2023-06.  Different versions are stored under `s3://ai2-llm/pretraining-data/sources/common-crawl/`:

### v0

Sharded output of CCNet pipline. Duplicate paragraphs removed (exact match, but only comparing against a ~2% sample of paragraphs in the corpus). Bucketed by language (fasttext), and English perplexity on wikipedia-trained 5-gram language model.

### v1

Post-process of v0. Drop non-English documents. Deduplicate whole documents by URL. Coalesce shards.

~4.8T tokens. High/Med/Low quality split: 20%/25%/55%

**v1-small** is an 8.5% sample of `v1`, about 300B tokens.

### v2

Post-process of v1. Remove duplicate paragraphs across the entire corpus

## CCNet Overview

We run a fork of CCNet at https://github.com/allenai/cc_net.git

We are not storing our own copy of the CC data. Files are downloaded on demand from http://data.commoncrawl.org. The Hashes and Transform steps each involve a full pass over the data. Downloads can be throttled, which is the main cause of failure.

The pipeline has two fundamental steps:

**Hash Content**

A sha hash is computed for each paragraph of each document.
This step is unlikely to ever change, so we pre-compute it and store the data S3 under `raw`.

**Deduplicate and Transform**

Use the shas to deduplicate the data within a single dump at the paragraph level. Produce separate sets of JSON files for each shard, in AI2 format.  Data is uploaded to S3 when finished.

## Running CCNet

Create an EC2 machine from [this template](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#LaunchTemplateDetails:launchTemplateId=lt-0be9ec34ba9794d3e).
The user-data initialization script will install dependencies and place a `READY` file in the home directory when finished. Check `/var/log/cloud-init-output.log` for progress.

Run commands from `$HOME/cc_net`. All commands are idempotent. Sometimes a shard will fail because the CC download was unsuccessful. In that case, repeat the command until it succeeds.

### Hash Content

This step is CPU-bound, so is most efficiently run on a low-memory-per-CPU machine, like a `c6a.32xlarge`

```
make hashes dump=<YYYY-nn> threads=<# CPUs>

```
Takes ~1200 CPU hours

When complete, each shard should have 1600 files uploaded to S3 under `raw/hashes`. Check status of pre-computed hashes with this command:

```
for d in `aws s3 ls s3://ai2-llm/pretraining-data/sources/common-crawl/raw/hashes/ | tr -s ' ' | cut -d' ' -f 3` ; do echo $d has `aws s3 ls s3://ai2-llm/pretraining-data/sources/common-crawl/raw/hashes/$d | wc -l`; done
```

### Deduplicate and Transform

This step is the slowest, and memory-bound, so it's most efficiently run on a high-memory-per-CPU machine, like a `u-3tb1.56xlarge`.
```
make transform dump=<YYYY-nn> threads=<available RAM / 20GB>
```
Takes ~5000 CPU hours

Check status of transformed data with this command:
```
for d in `aws s3 ls s3://ai2-llm/pretraining-data/sources/common-crawl/v0/documents/mined_split/ | tr -s ' ' | cut -d' ' -f 3` ; do echo $d has `aws s3 ls s3://ai2-llm/pretraining-data/sources/common-crawl/v0/documents/mined_split/$d | wc -l`; done
```

## Troubleshooting

Look in `cc_net/data/logs` for the logs of sub-processes that handle the individual tasks. A `Failed job ... has not produced any output` error indicates that the process was killed, probably for running out of memory.

## Dumps Processed

Our dataset includes data from the following dumps:

| dump    |
|---------|
| 2023-06 |
| 2022-49 |
| 2022-40 |
| 2022-33 |
| 2022-27 |
| 2022-21 |
| 2022-05 |
| 2021-49 |
| 2021-43 |
| 2021-39 |
| 2021-31 |
| 2021-25 |
| 2021-21 |
| 2021-17 |
| 2021-10 |
| 2021-04 |
| 2020-50 |
| 2020-45 |
| 2020-40 |
| 2020-34 |
| 2020-29 |
| 2020-24 |
| 2020-16 |
| 2020-10 |
| 2020-05 |
