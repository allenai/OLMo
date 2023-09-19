#!/bin/bash
dataset_names="ade_corpus_v2 banking_77 neurips_impact_statement_risks one_stop_english overruling semiconductor_org_types systematic_review_inclusion tai_safety_research terms_of_service tweet_eval_hate twitter_complaints"

for dataset_name in $dataset_names; do
    echo $dataset_name
    python process_data.py --dataset_path "ought/raft" --dataset_name $dataset_name --split test --output_folder /home/haop/datasets/
done
