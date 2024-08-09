#!/usr/bin/env bash

RUN_NAME="index_v4_dolma-v1_7_olmo_1-2"

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
    "
