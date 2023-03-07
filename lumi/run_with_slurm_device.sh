#!/bin/bash

export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*
