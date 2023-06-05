# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
FILENAMES_FILE=$1
VERSION=$2
OUTDIR=$3

echo $(wc -l $FILENAMES_FILE)
cat $FILENAMES_FILE| parallel -j 15 --bar --max-args=1 python get_num_tokens.py {1} $VERSION $OUTDIR
