# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
FILENAMES_FILE=$1
VERSION=$2
LANG_LIST=$3

echo $(wc -l $FILENAMES_FILE)
cat $FILENAMES_FILE| parallel -j 20 --bar --max-args=1 python run_pii.py --filename {1} --new-version $VERSION --lang-list $LANG_LIST
