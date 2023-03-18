URL=$1
LOCAL_DIR=$2

S3_LOCATION="s3://ai2-llm/pretraining-data/sources/stack-dedup/raw"

DIR="$(dirname $1)"
LANG="$(basename $DIR)"
FILENAME="$(basename $URL)"
JSONL_FILENAME=${FILENAME%.*}.jsonl.gz

SOURCE_PATH=$LOCAL_DIR/$LANG/$JSONL_FILENAME
DEST_PATH=$S3_LOCATION/$LANG/$JSONL_FILENAME

# DEST_EXISTS=$(aws s3 ls $dest_path)

DEST_SIZE=$(aws s3 ls --summarize --human-readable $DEST_PATH | tail -1)
EMPTY_SIZE_STR=': 0 Bytes'

if [[ "$DEST_SIZE" == *"$EMPTY_SIZE_STR"* ]]
then
    # echo $SOURCE_PATH $DEST_PATH

    # Load the remote URL and save it as a jsonl.gz file under LOCAL_DIR
    python download_url.py $URL $LOCAL_DIR

    # Copy saved jsonl.gz to S3 bucket.
    aws s3 cp --quiet $SOURCE_PATH $DEST_PATH

    # Remove the file from LOCAL_DIR to avoid filling up the disk space.
    rm $SOURCE_PATH
fi
