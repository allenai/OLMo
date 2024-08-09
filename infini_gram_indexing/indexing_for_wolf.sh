#!/usr/bin/env bash

RUN_NAME="index_v4_dolma-v1_7_olmo"

gantry run \
  --allow-dirty \
  --name ${RUN_NAME} \
  --task-name ${RUN_NAME} \
  --description ${RUN_NAME} \
  --workspace ai2/hb-wolf-olmo \
  --budget ai2/oe-training \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --preemptible \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --cpus 186 \
  --memory 1912GiB \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "cd infini_gram_indexing ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/v1_6-decontaminated/books/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/v1_6-decontaminated/pes2o/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/v1_6-decontaminated/wiki/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/v1_6-decontaminated/wiki/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/megawika/v1/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/redpajama_v1_decon_fix/stackexchange/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/redpajama_v1_decon_fix/arxiv/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/tulu_flan/v2-decontaminated-60M-shots_all-upweight_1-dialog_false-sep_newline/train/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/cc-news/v3/gpt-neox-olmo-dolma-v1_5/*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/0 ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-0*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-1*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-2*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-3*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-40*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-41*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-42*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-43*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-44*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-45*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/1 ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-46*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-47*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5/part-48*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/c4/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix/gpt-neox-olmo-dolma-v1_5/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/reddit/v5-dedupe-pii-nsfw-toxic-fuzzydd-length/gpt-neox-olmo-dolma-v1_5/*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/2 ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-00*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-01*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-02*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-03*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-04*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-05*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-06*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-07*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-08*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-090*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-091*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-092*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-093*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/3 ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-094*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-095*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-096*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-097*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-098*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-099*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/gpt-neox-olmo-dolma-v1_5/part-1*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/4 ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001/cc_en_head/gpt-neox-olmo-dolma-v1_5/*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/5 ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001/cc_en_middle/gpt-neox-olmo-dolma-v1_5/*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/6 ; \
    python indexing_for_wolf.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001/cc_en_tail/gpt-neox-olmo-dolma-v1_5/*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v4_dolma-v1_7_olmo/7 ; \
    "
