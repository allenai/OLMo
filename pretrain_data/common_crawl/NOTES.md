# Preparing CommonCrawl Data

## Overview

We run a fork of CCNet at https://github.com/allenai/cc_net.git

The pipeline has three fundamental steps:

**Hash Content**

A sha hash is computed for each paragraph of each document.
This step is unlikely to ever change, so we pre-compute it and store the data S3 under `raw`.
The step is CPU-bound, so is most efficiently run on a low-memory-per-CPU machine, like a `c6a.32xlarge`

**Deduplicate and Transform**

Use the shas to deduplicate the data within a single dump at the paragraph level. Produce separate sets of JSON files for each shard.
This step is the slowest, and memory-bound, so it's most efficiently run on a high-memory-per-CPU machine, like a `u-3tb1.56xlarge`. Data remains on local disk. 

**Regroup**

Consolidate data from all the shards into a set of files of fixed size for each language/perplexity bucket. Data is uploaded to S3 under `documents`. The fastest step, bound by CPU (or disk IO?).

Note that we are not storing our own copy of the CC data. Files are downloaded on demand from http://data.commoncrawl.org. The Hashes and Transform steps each involve a full pass over the data.

## Running

Create an EC2 machine from [this template](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#LaunchTemplateDetails:launchTemplateId=lt-0be9ec34ba9794d3e).
The user-data initialization script will install dependencies and place a `READY` file in the home directory when finished. Check `/var/log/cloud-init-output.log` for progress.

Run commands from `$HOME/cc_net`. All commands are idempotent. Sometimes a shard will fail because the CC download was unsuccessful. In that case, repeat the command until it succeeds.

### Hash Content
```
make hashes dump=<YYYY-nn> threads=<# CPUs>

```
Takes ~1200 CPU hours

When complete, each shard should have 1600 files uploaded to S3 under `raw/hashes`. Check status of pre-computed hashes with this command:

```
for d in `aws s3 ls s3://ai2-llm/pretraining-data/sources/common-crawl/raw/hashes/ | tr -s ' ' | cut -d' ' -f 3` ; do echo $d has `aws s3 ls s3://ai2-llm/pretraining-data/sources/common-crawl/raw/hashes/$d | wc -l`; done
```

### Deduplicate and Transform
```
make transform dump=<YYYY-nn> threads=<available RAM / 20GB>
```
Takes ~5000 CPU hours

### Regroup
```
make regroup dump=<YYYY-nn> threads=<# CPUs>
```
Takes ~200 CPU hours

Check output before terminating the instance:

```
aws s3 ls --recursive s3://ai2-llm/pretraining-data/sources/common-crawl/documents/v1/<YYYY-nn>
```

## Troubleshooting

Look in `cc_net/data/logs` for the logs of sub-processes that handle the individual tasks. A `Failed job ... has not produced any output` error indicates that the process was killed, probably for running out of memory.

## Status

|dump|hashes|documents|
|---|---|---|
|2023-14| | |
|2023-06|X| |
|2022-49|X| |
|2022-40|X| |
|2022-33|X| |
|2022-27|X| |
|2022-21|X| |
|2022-05|X| |
|2021-49|X| |
|2021-43|X| |
|2021-39|X| |
|2021-31|X| |
|2021-25|.| |
|2021-21|.| |
|2021-17|.| |
|2021-10|.| |
|2021-04| | |
|2020-50|.| |
|2020-45| | |
|2020-40| | |
|2020-34| | |
|2020-29| | |
|2020-24| | |
|2020-16| | |
|2020-10| | |
|2020-05| | |
|2019-51| | |
|2019-47| | |
|2019-43| | |
|2019-39| | |
|2019-35| | |
|2019-30| | |
|2019-26| | |
|2019-22| | |
|2019-18| | |
|2019-13| | |
|2019-09| | |
|2019-04| | |
|2018-51| | |
|2018-47| | |
|2018-43| | |
|2018-39| | |
|2018-34| | |
|2018-30| | |
|2018-26| | |
|2018-22| | |
|2018-17| | |
|2018-13| | |
|2018-09| | |
|2018-05| | |
|2017-51| | |
|2017-47| | |
|2017-43| | |
|2017-39| | |
|2017-34| | |
|2017-30| | |
|2017-26| | |
|2017-22| | |
|2017-17| | |
|2017-13| | |
|2017-09| | |
|2017-04| | |
|2016-50| | |
|2016-44| | |
|2016-40| | |
|2016-36| | |
|2016-30| | |
|2016-26| | |
|2016-22| | |
|2016-18| | |
|2016-07| | |
|2015-48| | |
|2015-40| | |
|2015-35| | |
|2015-32| | |
|2015-27| | |
|2015-22| | |
|2015-18| | |
|2015-14| | |
|2015-11| | |
|2015-06| | |
|2014-52| | |
|2014-49| | |
|2014-42| | |
|2014-41| | |
|2014-35| | |
|2014-23| | |
|2014-15| | |
|2014-10| | |
|2013-48| | |
|2013-20| | |
