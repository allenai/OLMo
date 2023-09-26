perplexity_dir=$1

# get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $perplexity_dir/v3_small/

# backwards compatability for v2
cp -r $perplexity_dir/v2_small/c4_en $perplexity_dir/v3_small/c4_en 
cp -r $perplexity_dir/v2_small/pile $perplexity_dir/v3_small/pile
cp -r $perplexity_dir/v2_small/m2d2_s2orc $perplexity_dir/v3_small/m2d2_s2orc
cp -r $perplexity_dir/v2_small/wikitext_103 $perplexity_dir/v3_small/wikitext_103
cp -r $perplexity_dir/v2_small/ice $perplexity_dir/v3_small/ice

# new data
mkdir -p $perplexity_dir/v3_small/dolma_books/test
mkdir -p $perplexity_dir/v3_small/dolma_books/val
cp $perplexity_dir/v3/dolma-v1_5/test/test_books.jsonl.gz $perplexity_dir/v3_small/dolma_books/test
cp $perplexity_dir/v3/dolma-v1_5/val/val_books.jsonl.gz $perplexity_dir/v3_small/dolma_books/val

mkdir -p $perplexity_dir/v3_small/dolma_common-crawl/test
mkdir -p $perplexity_dir/v3_small/dolma_common-crawl/val
cp $perplexity_dir/v3/dolma-v1_5/test/test_common-crawl.jsonl.gz $perplexity_dir/v3_small/dolma_common-crawl/test
cp $perplexity_dir/v3/dolma-v1_5/val/val_common-crawl.jsonl.gz $perplexity_dir/v3_small/dolma_common-crawl/val

mkdir -p $perplexity_dir/v3_small/dolma_pes2o/test
mkdir -p $perplexity_dir/v3_small/dolma_pes2o/val
cp $perplexity_dir/v3/dolma-v1_5/test/test_pes2o.jsonl.gz $perplexity_dir/v3_small/dolma_pes2o/test
cp $perplexity_dir/v3/dolma-v1_5/val/val_pes2o.jsonl.gz $perplexity_dir/v3_small/dolma_pes2o/val

mkdir -p $perplexity_dir/v3_small/dolma_reddit/test
mkdir -p $perplexity_dir/v3_small/dolma_reddit/val
cp $perplexity_dir/v3/dolma-v1_5/test/test_reddit_uniform.jsonl.gz $perplexity_dir/v3_small/dolma_reddit/test
cp $perplexity_dir/v3/dolma-v1_5/val/val_reddit_uniform.jsonl.gz $perplexity_dir/v3_small/dolma_reddit/val

mkdir -p $perplexity_dir/v3_small/dolma_wiki/test
mkdir -p $perplexity_dir/v3_small/dolma_wiki/val
cp $perplexity_dir/v3/dolma-v1_5/test/test_wiki.jsonl.gz $perplexity_dir/v3_small/dolma_wiki/test
cp $perplexity_dir/v3/dolma-v1_5/val/val_wiki.jsonl.gz $perplexity_dir/v3_small/dolma_wiki/val

mkdir -p $perplexity_dir/v3_small/dolma_stack/test
mkdir -p $perplexity_dir/v3_small/dolma_stack/val
cp $perplexity_dir/v3_not_deconned/dolma-v1_5/test/test_stack_uniform.jsonl.gz $perplexity_dir/v3_small/dolma_stack/test
cp $perplexity_dir/v3_not_deconned/dolma-v1_5/val/val_stack_uniform.jsonl.gz $perplexity_dir/v3_small/dolma_stack/val

    