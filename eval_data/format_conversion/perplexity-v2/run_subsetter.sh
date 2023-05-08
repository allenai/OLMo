output_dir=$1
datasets_name=$2
token_quota=$3
test_and_val=$4
subdomain_regex=$5
filename_prefix=$6
batch_size=$7
subdomain_from_metadata=$8

# get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# make dir for datasets_name if not already existing
mkdir -p $output_dir/$datasets_name
mkdir -p $output_dir/stats

test_and_val_flag=""
# if test_and_val is true, then
if [ "$test_and_val" = true ] ; then
    # make dir for datasets_name if not already existing
    mkdir -p $output_dir/$datasets_name/test
    mkdir -p $output_dir/$datasets_name/val

    test_and_val_flag="--test_and_val"
    test_and_val=""

else
    mkdir -p $output_dir/$datasets_name/$test_and_val
fi

# if subdomain_regex is not false, then set subdomain_regex_flag
subdomain_regex_flag=""
if [ "$subdomain_regex" != false ] ; then
    subdomain_regex_flag="--subdomain_from_filename_regex $subdomain_regex"
fi

# if subdomain_from_metadata is not false, then set subdomain_regex_flag
subdomain_from_metadata_flag=""
if [ "$subdomain_from_metadata" != false ] ; then
    subdomain_from_metadata_flag="--subdomain_from_metadata $subdomain_from_metadata"
fi

# if filename_prefix is false, then just use datasets_name
if [ "$filename_prefix" = false ] ; then
    filename_prefix=$datasets_name
fi

# if token_quota is not false, then set token_quota_flag
token_quota_flag=""
if [ "$token_quota" != false ] ; then
    token_quota_flag="--token_quotas $token_quota"
fi

# rest of the args are the shards
shards=${@:9}

time python $SCRIPT_DIR/subsetter.py \
    $token_quota_flag \
    $test_and_val_flag \
    --data_files $shards \
    --batch_sizes $batch_size \
    --output_dir $output_dir/$datasets_name/$test_and_val \
    --stats_output_file $output_dir/stats/${filename_prefix}_${test_and_val}_stats.json \
    --file_prefix ${filename_prefix} \
    $subdomain_regex_flag \
    $subdomain_from_metadata_flag