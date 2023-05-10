# This script analyses The Stack (dedup) to produce some statistics.
# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
STAT_FILES=$1

ls -l $STAT_FILES/*/* | awk '{print $9}'| parallel -j 20 --bar --max-args=1 python min_max_tokens.py {1}

