out_dir=$1
in_dir=$2

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# some datasets to small to support batching at this step, they are just run with batch size 1 below
batch_size=1 # don't truncate from batching

bash $SCRIPT_DIR/run_subsetter.sh $out_dir c4_en 1000000 true false false $batch_size false $in_dir/c4_en/val/*.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir mc4_en 1000000 true false false $batch_size false $in_dir/mc4/val/*.gz

for split in val test
do
    echo starting $split
    total_files=$(ls $in_dir/c4_100_domains/$split/*.gz | wc -l)
    files_done=0
    for file in $in_dir/c4_100_domains/$split/*.gz
    do
        echo $files_done / $total_files
        files_done=$((files_done+1))
        # remove .json.gz.<split>.json.gz from filename
        filename=$(basename -- "$file")
        subdomain="${filename%\.json\.gz\.*\.json\.gz}"
        bash $SCRIPT_DIR/run_subsetter.sh $out_dir c4_100_domains 100000 $split '(.*?)\.json\.gz\.(test|val)\.json\.gz' c4_100_domains_$subdomain 1 false $file
    done
done

bash $SCRIPT_DIR/run_subsetter.sh $out_dir 4chan 1000000 true false false $batch_size false $in_dir/4chan/*.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir gab 1000000 true false false $batch_size false $in_dir/gab/*.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir manosphere 1000000 true '^(.+)\.jsonl\.gz$' false $batch_size false $in_dir/manosphere/*.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir pile 10000000 val false false $batch_size 'pile_set_name' $in_dir/pile/val/*.gz
bash $SCRIPT_DIR/run_subsetter.sh $out_dir pile 10000000 test false false $batch_size 'pile_set_name' $in_dir/pile/test/*.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir m2d2_s2orc 10000000 val '^(.+)(?:_test|_valid)\.jsonl\.gz$' false $batch_size false $in_dir/m2d2/s2orc_paragraphs/val/*.gz
bash $SCRIPT_DIR/run_subsetter.sh $out_dir m2d2_s2orc 10000000 test '^(.+)(?:_test|_valid)\.jsonl\.gz$' false $batch_size false $in_dir/m2d2/s2orc_paragraphs/test/*.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir m2d2_wiki 10000000 val '^(.+)(?:_test|_valid)\.jsonl\.gz$' false $batch_size false $in_dir/m2d2/wikipedia/val/*.gz
bash $SCRIPT_DIR/run_subsetter.sh $out_dir m2d2_wiki 10000000 test '^(.+)(?:_test|_valid)\.jsonl\.gz$' false $batch_size false $in_dir/m2d2/wikipedia/test/*.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir ice false true '^(.+)\.jsonl\.gz$' false $batch_size false $in_dir/ice/*.gz

for file in $in_dir/twitterAAE_helm/*.gz
do
    # remove extension from filename
    filename=$(basename -- "$file")
    subdomain="${filename%\.jsonl\.gz}"
    bash $SCRIPT_DIR/run_subsetter.sh $out_dir twitterAEE false true '(.*?)\.jsonl\.gz' twitterAEE_$subdomain $batch_size false $file
done

bash $SCRIPT_DIR/run_subsetter.sh $out_dir wikitext_103 false val false false 1 false $in_dir/wikitext_103/wiki.valid.jsonl.gz
bash $SCRIPT_DIR/run_subsetter.sh $out_dir wikitext_103 false test false false 1 false $in_dir/wikitext_103/wiki.test.jsonl.gz

bash $SCRIPT_DIR/run_subsetter.sh $out_dir ptb false val false false 1 false $in_dir/ptb/ptb.valid.jsonl.gz
bash $SCRIPT_DIR/run_subsetter.sh $out_dir ptb false test false false 1 false $in_dir/ptb/ptb.test.jsonl.gz