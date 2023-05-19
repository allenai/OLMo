output_dir=$1
datasets_name=$2
token_quota=$3
test_and_val=$4

# get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# make dir for datasets_name if not already existing
mkdir -p $output_dir/$datasets_name

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

# rest of the args are the shards
shards=${@:4}

time python $SCRIPT_DIR/subsetter.py --token_quotas $token_quota $test_and_val_flag --data_files $shards --batch_sizes 256 --output_dir $output_dir/$datasets_name/$test_and_val --stats_output_file $output_dir/${datasets_name}_${test_and_val}_stats.json --file_prefix ${datasets_name}