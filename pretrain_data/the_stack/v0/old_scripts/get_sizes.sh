S3_LOCATION="s3://ai2-llm/pretraining-data/sources/stack-dedup/raw"

cat "$1" | while read LANG; do
	#echo "$LANG"
	LANG_PATH=$S3_LOCATION/$LANG/
	DEST_SIZE=$(aws s3 ls --summarize $LANG_PATH | tail -1)
	echo $LANG "\t" $DEST_SIZE
done

