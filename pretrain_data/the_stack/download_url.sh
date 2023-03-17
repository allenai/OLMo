dir="$(dirname $1)"
dir="$(basename $dir)"
filename="$(basename $1)"
source_name=${filename%.*}.jsonl.gz

source_path=$2/$dir/$source_name
dest_path="s3://ai2-llm/pretraining-data/sources/stack-dedup/raw/$dir/$source_name"

dest_exists=$(aws s3 ls $dest_path)

if [ -z "$dest_exists" ]
then
    #echo $source_path $dest_path
    python download_url.py $1 $2

    aws s3 cp --quiet $source_path $dest_path
    rm $source_path
fi

