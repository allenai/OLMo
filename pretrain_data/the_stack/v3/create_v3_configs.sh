# GNU-parallel used for parallel processing.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.

LANG_LIST=$1
OUTPUT_DIR=$2

cat $LANG_LIST| parallel -j 1 --bar --max-args=1 python create_v3_configs.py {1} $OUTPUT_DIR
