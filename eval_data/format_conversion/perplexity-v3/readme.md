To reproduce v3

```
export DATA_DIR=<path to eval-data/perplexity part of bucket data>
```

```
bash run_subsetter.sh $DATA_DIR
```

for m2d2
```
cd $DATA_DIR/raw/m2s2/
mkdir s2orc
parallel --eta --bar "tar -xzf {} -C s2orc" ::: s2orc_zips/*.gz
```