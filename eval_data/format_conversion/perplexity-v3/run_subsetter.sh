perplexity_dir=$1

# get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/raw/twitterAAE_HELM_fixed/*.gz \
    --output_dir $perplexity_dir/v3/twitterAAE_HELM_fixed \
    --seed 42 \
    --sample_evenly_by_file

python $SCRIPT_DIR/m2d2_subsetter.py \
    --input_files $perplexity_dir/raw/m2d2/s2orc/*/valid.txt \
    --output_dir $perplexity_dir/v3/m2d2_s2orc_unsplit/val \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --tokens_per_subdomain 100000

python $SCRIPT_DIR/m2d2_subsetter.py \
    --input_files $perplexity_dir/raw/m2d2/s2orc/*/test.txt \
    --output_dir $perplexity_dir/v3/m2d2_s2orc_unsplit/test \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --tokens_per_subdomain 100000

python $SCRIPT_DIR/m2d2_subsetter.py \
    --input_files $perplexity_dir/raw/m2d2/wikipedia/*/valid.txt \
    --output_dir $perplexity_dir/v3/m2d2_wikipedia_unsplit/val \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --tokens_per_subdomain 100000

python $SCRIPT_DIR/m2d2_subsetter.py \
    --input_files $perplexity_dir/raw/m2d2/wikipedia/*/test.txt \
    --output_dir $perplexity_dir/v3/m2d2_wikipedia_unsplit/test \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --tokens_per_subdomain 100000

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/v0/4chan_meta_sep/*.jsonl.gz \
    --output_dir $perplexity_dir/v3/4chan_meta_sep \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 1000000

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/v0/manosphere_meta_sep/*.jsonl.gz \
    --output_dir $perplexity_dir/v3/manosphere_meta_sep \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 1000000

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/v0/ice_fixed/*.gz \
    --output_dir $perplexity_dir/v3/ice_fixed \
    --seed 42 \
    --sample_evenly_by_file

python $SCRIPT_DIR/subsetter.py \
    --split_names val \
    --input_files $perplexity_dir/raw/redpajama/v1/documents/split=valid/dataset=*/*.gz \
    --output_dir $perplexity_dir/v3/redpajama \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 700000 \
    --source_has_subdomain \
    --sample_evenly_by_subdomain

python $SCRIPT_DIR/subsetter.py \
    --split_names test \
    --input_files $perplexity_dir/raw/redpajama/v1/documents/split=test/dataset=*/*.gz \
    --output_dir $perplexity_dir/v3/redpajama \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 700000 \
    --source_has_subdomain \
    --sample_evenly_by_subdomain

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/raw/falcon-refinedweb/v0-0.05-heldout/documents/*.gz \
    --output_dir $perplexity_dir/v3/falcon-refinedweb \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 1000000


mkdir -p $perplexity_dir/v3/wikitext_103/val
mkdir -p $perplexity_dir/v3/wikitext_103/test
cp $perplexity_dir/v0/wikitext_103/wiki.valid.jsonl.gz $perplexity_dir/v3/wikitext_103/val/val.jsonl.gz
cp $perplexity_dir/v0/wikitext_103/wiki.test.jsonl.gz $perplexity_dir/v3/wikitext_103/test/test.jsonl.gz

mkdir -p $perplexity_dir/v3/ptb/val
mkdir -p $perplexity_dir/v3/ptb/test
cp $perplexity_dir/v0/ptb/ptb.valid.jsonl.gz $perplexity_dir/v3/ptb/val/val.jsonl.gz
cp $perplexity_dir/v0/ptb/ptb.test.jsonl.gz $perplexity_dir/v3/ptb/test/test.jsonl.gz

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/v0/c4_en/val/*.gz \
    --output_dir $perplexity_dir/v3/c4_en \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 1000000

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/v0/mc4/val/*.gz \
    --output_dir $perplexity_dir/v3/mc4 \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 1000000

python $SCRIPT_DIR/subsetter.py \
    --split_names val \
    --input_files $perplexity_dir/v0/pile/val/*.gz \
    --output_dir $perplexity_dir/v3/pile \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 2200000 \
    --pile_subdomain_format \
    --sample_evenly_by_subdomain

python $SCRIPT_DIR/subsetter.py \
    --split_names test \
    --input_files $perplexity_dir/v0/pile/test/*.gz \
    --output_dir $perplexity_dir/v3/pile \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 2200000 \
    --pile_subdomain_format \
    --sample_evenly_by_subdomain

python $SCRIPT_DIR/subsetter.py \
    --input_files $perplexity_dir/v0/gab/*.gz \
    --output_dir $perplexity_dir/v3/gab \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 1000000


python $SCRIPT_DIR/subsetter.py \
    --split_names val \
    --input_files $perplexity_dir/v0/c4_100_domains/val/*.gz \
    --output_dir $perplexity_dir/v3/c4_100_domains \
    --seed 42 \
    --sample_evenly_by_file \
    --split_token_count_target 10000000 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --subdomain_from_file_name_minus_extension .json.gz.val.json.gz

python $SCRIPT_DIR/subsetter.py \
    --split_names test \
    --input_files $perplexity_dir/v0/c4_100_domains/test/*.gz \
    --output_dir $perplexity_dir/v3/c4_100_domains \
    --seed 42 \
    --sample_evenly_by_file \
    --split_token_count_target 10000000 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --subdomain_from_file_name_minus_extension .json.gz.test.json.gz


python $SCRIPT_DIR/subsetter.py \
    --input_files v0/dolma-v1_5/val/*.gz \
    --output_dir $perplexity_dir/v3/dolma-v1_5 \
    --seed 42 \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --split_token_count_target 2000000 \
    --sample_evenly_by_file 

python $SCRIPT_DIR/subsetter.py \
    --split_names val \
    --input_files $perplexity_dir/v0/reddit/val_reddit.jsonl.gz \
    --output_dir $perplexity_dir/v3/reddit \
    --seed 42 \
    --sample_evenly_by_file

python $SCRIPT_DIR/subsetter.py \
    --split_names test \
    --input_files $perplexity_dir/v0/reddit/test_reddit.jsonl.gz \
    --output_dir $perplexity_dir/v3/reddit \
    --seed 42 \
    --sample_evenly_by_file
