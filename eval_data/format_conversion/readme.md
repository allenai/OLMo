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

Please store the results in `s3://ai2-llm/eval-data/perplexity/` or `s3://ai2-llm/eval-data/downstream` with directories for `val`, and `test` splits if present. For example the pile is in `s3://ai2-llm/eval-data/perplexity/pile/val/` and `s3://ai2-llm/eval-data/perplexity/pile/test/`. Please store files as `.jsonl.gz` to save space.