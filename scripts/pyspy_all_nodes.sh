#!/bin/bash

set -euxo pipefail

for i in $(scontrol show hostnames $(scontrol show job $1 | grep -oP ' NodeList=\K\S+') | sort); do
  echo Hostname: $i
  srun --interactive --jobid=$1 -w $i \
    singularity exec \
      -B"$PROJECT_DIR:$PROJECT_DIR" \
      -B"$SCRATCH_DIR:$SCRATCH_DIR" \
      -B"$FLASH_DIR:$FLASH_DIR" \
      -B /opt/cray:/opt/cray \
      -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
      -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
      $PROJECT_DIR/containers/llm-lumi_latest.sif \
        bash scripts/pyspy_all_processes.sh
  echo
done
