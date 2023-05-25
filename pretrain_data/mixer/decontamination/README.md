# Decontamination Runbook

This runbook explains how the default decontamination filter is created.

We use the following rules to create a decontamination filter:

- paragraph level decontamination: we use the bloom filter to figure out which paragraphs to toss out;
paragraph is the smallest unit of decontamination
- ngram filtering: for purposes of building the tokenizer, we only keep spans of text that have at least
13 `uniseg` tokens.

## Step 1: Tag the Data by Length

We want to get the count of `uniseg` tokens in each paragraph. To do so, we run the following command:

```shell
ai2_llm_filters \
    -d ian_perplexity_test_valid_set/v2 \
    -n counts \
    -t uniseg_length_paragraphs_v1 \
    -p 64  \
    --skip-on-failure
```

The `uniseg_length_paragraphs_v1` returns the count of `uniseg` tokens in each paragraph, whitepsace excluded.
This is the same as how [BFF does it](https://github.com/allenai/bff/blob/fcc0d5799120a35424f1a4433dfea89bad77ae3e/src/main.rs#L102-L111).

## Step 2: Run Mixer

We run mixer to remove paragraphs that are shorter than 13 `uniseg` tokens.

```shell
MIXER_BIN="pretrain_data/mixer/target/release/mixer"
$MIXER_BIN \
    pretrain_data/mixer/decontamination/mixer_decontamination_config.json
```

## Step 3: Run Deduper to train a bloom filter

We run the deduper to train a bloom filter. Ablations will use the bloom filter to remove paragraphs.

```shell
DEDUPER_BIN="pretrain_data/mixer/target/release/deduper"
$DEDUPER_BIN \
    pretrain_data/mixer/decontamination/deduper_decontamination_config.json
```
This will save the bloom filter to `/tmp/deduper_decontamination_lucas_20230523.bin`.

Note how the deduper needs the count of paragraphs in the input config; to get that, we used the following

```bash
aws s3 cp s3://ai2-llm/pretraining-data/sources/ian_perplexity_test_valid_set/v2_filtered/documents/ian_perplexity_test_valid_set-0000.json.gz .

zcat ian_perplexity_test_valid_set-0000.json.gz \
    | jq '.attributes.counts__uniseg_length_paragraphs_v1__paragraph | select(.[2] >= 13) | length' -c \
    | tqdm \
    | python -c 'import sys; print(sum(int(e) for e in sys.stdin))'
```

## Step 4: Upload the Bloom Filter to S3

We upload the bloom filter to S3:

```shell
aws s3 cp \
    /tmp/deduper_decontamination_lucas_20230523.bin \
    s3://ai2-llm/eval-data/perplexity/blocklists/eval_subset_v2/deduper_decontamination_lucas_20230522.bin
```
