#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift

LOAD_STEP=$1
shift

ANNEAL_STEPS=$1
shift

for LR in 1.56e-2
do
  ./scripts/augusta/beaker/peteish1-wsd-anneal-launch.sh $NUM_NODES $LR $LOAD_STEP $ANNEAL_STEPS ${@}
done