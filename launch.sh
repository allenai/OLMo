#! /usr/bin/env bash

CONFIG_PATH=configs/road-to-1_7/runs/r70b-baseline-sources-1b-150b.yaml sbatch -J r70b-S --nodes=32 --partition dev-g --time=3:00:00 scripts/lumi/olmo-small-ablation-on-lumi.sh --time_limit=10200 --global_train_batch_size=2048 --device_train_microbatch_size=8 --data.num_workers=0 --data.prefetch_factor=16 --model.flash_attention=false --compile=null --save_folder='${path.choose:${oc.env:SCRATCH_DIR,no_exist}/checkpoints,/results}/${oc.env:SLURM_JOB_ID,${run_name}}'
