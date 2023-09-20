To reproduce v3

```
export DATA_DIR=<path to eval-data/perplexity part of bucket data>
```

for m2d2
```
cd $DATA_DIR/raw/m2s2/
mkdir s2orc
parallel --eta --bar "tar -xzf {} -C s2orc" ::: s2orc_zips/*.gz
```

rebuilding 4chan from raw with meta sep
```
cd $DATA_DIR/raw/4chan
parallel --eta --bar "python LLM/eval_data/format_conversion/eval_data_converter.py --in_dir ./ --out_dir $DATA_DIR/v0/4chan_meta_sep --filename {} --in_format four_chan --metadata_header_seperate_line" ::: 4chan_shard*
```

rebuilding manosphere from raw with meta sep
```
cd $DATA_DIR/raw/manosphere
parallel --eta --bar "python LLM/eval_data/format_conversion/eval_data_converter.py --in_dir ./ --out_dir $DATA_DIR/v0/manosphere_meta_sep --filename {} --in_format manosphere --metadata_header_seperate_line" ::: *
```

```
bash run_subsetter.sh $DATA_DIR
```

