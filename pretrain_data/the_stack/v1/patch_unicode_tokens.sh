# GNU-parallel used for parallel processing.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.

#LANG_LIST=$1
LANG_FILES=$1

cat $LANG_FILES| parallel -j 15 --bar --max-args=1 python patch_unicode_tokens.py {1}

