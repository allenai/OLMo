out_dir=$1
in_dir=$2

bash run_subsetter.sh $out_dir c4_en 1000000 true $in_dir/c4_en/val/*.gz

bash run_subsetter.sh $out_dir mc4_en 1000000 true $in_dir/mc4/val/*.gz

bash run_subsetter.sh $out_dir c4_100_domains 10000000 val $in_dir/c4_100_domains/val/*.gz
bash run_subsetter.sh $out_dir c4_100_domains 10000000 test $in_dir/c4_100_domains/test/*.gz

bash run_subsetter.sh $out_dir 4chan 1000000 true $in_dir/4chan/*.gz

bash run_subsetter.sh $out_dir gab 1000000 true $in_dir/gab/*.gz

bash run_subsetter.sh $out_dir manosphere 1000000 true $in_dir/manosphere/*.gz

bash run_subsetter.sh $out_dir pile 10000000 val $in_dir/pile/val/*.gz
bash run_subsetter.sh $out_dir pile 10000000 test $in_dir/pile/test/*.gz

bash run_subsetter.sh $out_dir m2d2_s2orc 10000000 val $in_dir/m2d2/s2orc_paragraphs/val/*.gz
bash run_subsetter.sh $out_dir m2d2_s2orc 10000000 test $in_dir/m2d2/s2orc_paragraphs/test/*.gz

bash run_subsetter.sh $out_dir m2d2_wiki 10000000 val $in_dir/m2d2/wikipedia/val/*.gz
bash run_subsetter.sh $out_dir m2d2_wiki 10000000 test $in_dir/m2d2/wikipedia/test/*.gz

bash run_subsetter.sh $out_dir ice 5000000 true $in_dir/ice/*.gz

bash run_subsetter.sh $out_dir twitterAAE 500000 true $in_dir/twitterAAE_helm/*.gz

mkdir $out_dir/wikitext_103/
mkdir $out_dir/wikitext_103/val
mkdir $out_dir/wikitext_103/test
cp $in_dir/wikitext_103/wiki.valid.jsonl.gz $out_dir/wikitext_103/val/
cp $in_dir/wikitext_103/wiki.test.jsonl.gz $out_dir/wikitext_103/test/