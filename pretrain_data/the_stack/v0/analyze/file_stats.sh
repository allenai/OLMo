# This script analyses The Stack (dedup) to produce some statistics.
# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
URL_LIST_FILE=$1
OUTPUT_DIR=$2
VERSION=$3

echo $(wc -l $URL_LIST_FILE)
cat $URL_LIST_FILE| parallel -j 10 --bar --max-args=1 python file_stats.py --url {1} --output-dir $OUTPUT_DIR --version $VERSION
