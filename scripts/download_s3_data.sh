file_name="part-0-00000.npy"
folder="eval-data/perplexity/v3_small_gptneox20b/wikitext_103/val"
remote_root="https://olmo-data.org"

mkdir -p ${folder}

wget ${remote_root}/${folder}/${file_name} -P ${folder}