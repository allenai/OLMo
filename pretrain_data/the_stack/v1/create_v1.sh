# GNU-parallel used for parallel processing.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.

LANG_FILES=$1
LOCAL_DIR=$2
MIXER=$3

cat $LANG_FILES| parallel -j 1 --bar --max-args=1 $MIXER $LOCAL_DIR/{1}.json

