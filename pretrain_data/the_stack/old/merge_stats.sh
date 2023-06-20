# GNU-parallel used for parallel processing.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.

LANG_LIST=$1
LANG_FILES=$2
LOCAL_DIR=$3
VERSION=$4

cat $LANG_LIST| parallel -j 10 --bar --max-args=1 ./merge_lang_stats.sh $LANG_FILES {1} $LOCAL_DIR $VERSION

