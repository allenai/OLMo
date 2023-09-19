perplexity_dir=$1

# get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/raw/twitterAAE_HELM_fixed/*.gz \
    --output_dir $perplexity_dir/v3/twitterAAE_HELM_fixed \
    --seed 42 \
    --sample_evenly_by_file