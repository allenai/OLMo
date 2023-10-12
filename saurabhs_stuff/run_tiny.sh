#!/bin/bash

export OMP_NUM_THREADS=8
run_name=c4-tiny-run-001-yes-act-logging-002
rm -rf "runs/${run_name}"
torchrun --nproc-per-node 2 scripts/train.py saurabhs_stuff/c4-tiny.yaml \
  --run_name="${run_name}" \
  --save_folder="runs/${run_name}" \
  --console_log_interval=1 