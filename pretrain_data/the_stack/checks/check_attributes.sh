# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
FILENAMES_FILE=$1

echo $(wc -l $FILENAMES_FILE)
cat $FILENAMES_FILE| parallel -j 10 --bar --max-args=1 ./check_file_exists.sh {1} attributes .jsonl v2 > missing_code_secrets.txt
