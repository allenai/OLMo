gantry run \
    --allow-dirty \
    --workspace ai2/cheap-decisions  \
    --priority normal \
    --gpus 0 \
    --preemptible \
    --cluster ai2/jupiter-cirrascale-2 \
    --budget ai2/oe-eval \
    --env-secret AWS_ACCESS_KEY_ID=JENA_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=JENA_AWS_SECRET_ACCESS_KEY \
    --shared-memory 10GiB \
    --weka=oe-eval-default:/data/input \
    --pip requirements.txt \
    --yes \
    -- /bin/bash -c "python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir 's3://ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001/step99000-unsharded' --destination-dir '/data/input/ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001/step99000-unsharded' --keep-olmo-artifacts --tokenizer 'olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json'"

#    --install install_torch.sh \
