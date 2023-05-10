# This script analyses The Stack (dedup) to produce some statistics.
# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
LANG_FILES_PATH=$1
LANG=$2
LOCAL_DIR=$3

S3_LOCATION="s3://ai2-llm/pretraining-data/sources/stack-dedup/v0/attributes/file_stats"

SOURCE_PATH=$LOCAL_DIR/merged_${LANG}_stats.tsv
DEST_PATH=$S3_LOCATION/$LANG/merged_${LANG}_stats.tsv

DEST_EXISTS=$(aws s3 ls $DEST_PATH)

DEST_SIZE=$(aws s3 ls --summarize --human-readable $DEST_PATH | tail -1)
EMPTY_SIZE_STR=': 0 Bytes'

# if [[ "$DEST_SIZE" = "" ]]
#if [[ "$DEST_SIZE" == *"$EMPTY_SIZE_STR"* ]]
#then
    # echo $SOURCE_PATH $DEST_PATH

    # Merge stats for a language under LOCAL_DIR
    python merge_lang_stats.py --lang-files-path $LANG_FILES_PATH --lang $LANG --output-path $SOURCE_PATH

    # Copy saved file to S3 bucket.
    aws s3 cp --quiet $SOURCE_PATH $DEST_PATH

    # Remove the file from LOCAL_DIR to avoid filling up the disk space.
    # rm $SOURCE_PATH
#fi

