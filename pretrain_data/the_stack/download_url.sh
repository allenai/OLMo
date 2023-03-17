dir="$(dirname $1)"
dir="$(basename $dir)"
filename="$(basename $1)"
source_name=${filename%.*}.jsonl.gz

source_path=$2/$dir/$source_name
dest_path="s3://ai2-llm/pretraining-data/sources/stack-dedup/raw/$dir/$source_name"

dest_exists=$(aws s3 ls $dest_path)

dest_size=$(aws s3 ls --summarize --human-readable $dest_path | tail -1)
empty_size=': 0 Bytes'
if [[ "$dest_size" == *"$empty_size"* ]]
then
    #echo $source_path $dest_path
    python download_url.py $1 $2

    aws s3 cp --quiet $source_path $dest_path
    rm $source_path
fi

