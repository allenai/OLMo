# This script analyses The Stack (dedup) to produce some statistics.
# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
LANG_LIST_FILE=$1
URL_LIST_FILE=$2
OUTPUT_FILE=$3

echo $(wc -l $LANG_LIST_FILE)
cat $LANG_LIST_FILE| parallel -j 10 --bar --max-args=1 python new_data_analysis.py --urls-file $URL_LIST_FILE --lang {1} --output-file infos/{1}_$OUTPUT_FILE

