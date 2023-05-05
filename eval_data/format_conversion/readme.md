The llm-eval team is building out scripts to convert evaluation data to a common format for the purposes of deduplication.

This format is line separated json for each eval example as follows:
```
{
    "source": "...",         # 'boolQ', 'pile', etc...
    "id": "...",             # source-specific identifier
    "text": "...",          # a single string joining all fields that should be decontaminated against
    "added": "...",          # timestamp ai2 acquired this data
    "metadata": {...}        # any source-specific metadata  that is worth retaining and not too large
}
```

You can add your converter as a static method of Formater in [eval_data_converter.py](https://github.com/allenai/LLM/blob/6d7d93818665a7142508cab552aa45268bb64a68/eval_data/format_conversion/eval_data_converter.py) if you haven't already implemented it separately. If you have many shard to convert you can invoke it in parallel like this:
```
parallel "python eval_data_converter.py --in_dir /path/to/data --out_dir /path/to/output --in_format dataset_format --filename {}" ::: *.jsonl.gz
```

In the following we will assume you have access to the LLM project bucket
```
$llm-bucket=<path to shared project bucket>
```

Please store the results in `$llm-bucket/eval-data/perplexity/` or `$llm-bucket/eval-data/downstream` with directories for `val`, and `test` splits if present. Also under the perplexity and downstream data dirs, subdirs `raw`, `v0`, `v1` and so on will track versioning. For example the pile is in `$llm-bucket/eval-data/perplexity/v0/pile/val/` and `$llm-bucket/eval-data/perplexity/v0/pile/test/`. Please store files as `.jsonl.gz` to save space.

## perplexity eval suite Raw to v0

Going from `$llm-bucket/eval-data/perplexity/raw/` to `$llm-bucket/eval-data/perplexity/v0` unpacks the data and standardizes to our format and adds some additional fields such as IDs where they are not already present. Use `eval_data_converter.py` with the appropreate `--in_format`, except for ICE which must be preprocessed following the steps in `LLM/eval_data/format_conversion/get_ice/readme.md`