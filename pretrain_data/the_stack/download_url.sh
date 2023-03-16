python preparation/stack/download_url.py $1 $2
dir="$(dirname $1)"
dir="$(basename $dir)"
filename="$(basename $1)"
source_name=${filename%.*}.jsonl.gz

source_path=$2/$dir/$source_name
dest_path="s3://ai2-llm/pretraining-data/sources/stack-dedup/raw/$dir/$source_name"

echo $source_path $dest_path
aws s3 cp $source_path $dest_path
# rm $2/$dir/$filename