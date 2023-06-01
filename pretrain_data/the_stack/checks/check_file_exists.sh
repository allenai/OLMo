FILENAME=$1
PREFIX=$2
POSTFIX=$3
VERSION=$4
S3_LOCATION="s3://ai2-llm/pretraining-data/sources/stack-dedup"

DEST_PATH=${S3_LOCATION}/${VERSION}/${PREFIX}/${FILENAME}${POSTFIX}

DEST_SIZE=$(aws s3 ls --summarize --human-readable $DEST_PATH | tail -1)
EMPTY_SIZE_STR=': 0 Bytes'

if [[ "$DEST_SIZE" == *"$EMPTY_SIZE_STR"* ]]
then
    echo $DEST_PATH
fi
