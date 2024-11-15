#!/bin/bash
# Run an interactive shell in our singularity image on LUMI.

singularity shell \
  -B"$PROJECT_DIR:$PROJECT_DIR" \
  -B"$SCRATCH_DIR:$SCRATCH_DIR" \
  -B"$FLASH_DIR:$FLASH_DIR" \
  -B /var/spool/slurmd,/opt/cray/,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/usr/lib64/libjson-c.so.3 \
  $OLMO_CONTAINER
