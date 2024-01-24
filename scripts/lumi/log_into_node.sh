#!/bin/bash

set -euxo pipefail

srun --interactive --pty --jobid=$1 \
  singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B /opt/cray:/opt/cray \
    -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $PROJECT_DIR/containers/llm-lumi-torch21_latest.sif \
      fish

set -euxo pipefail

srun --interactive --pty --jobid=5358908 \
  singularity shell \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B /opt/cray:/opt/cray \
    -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $PROJECT_DIR/containers/llm-lumi_latest.sif

ls | grep -v '\.txt' | grep -v -e 5329877 -e 5300701 | parallel tar -cvjSf {}.tar.bz2 {} ">" ~/logs/{}-tar.log
ls | grep -v '\.txt' | grep -v -e 5329877 -e 5300701 | xargs -I % tar -I "zstd -T0" -cf %.tar.zst % > ~/logs/tar-zst.log

ls -tr /scratch/project_462000229/checkpoints/ | head -n 33 | grep -v 5128986 | sed 's#^#/scratch/project_462000229/checkpoints/#' | parallel python scripts/storage_cleaner.py move --keep_src --append_wandb_path {} r2://olmo-checkpoints/ ">" ~/logs/{/.}-scratch-upload.log