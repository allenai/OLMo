#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift

LOAD_STEP=$1
shift

ANNEAL_STEPS=$1
shift

# for LR in 1.56e-2 7.81e-3 3.91e-3 1.95e-3 9.77e-4 4.88e-4 2.44e-4 1.22e-4 6.10e-5 3.05e-5 1.53e-5 7.63e-6
for LR in 3.91e-3 1.95e-3 9.77e-4 4.88e-4 2.44e-4 1.22e-4 6.10e-5 3.05e-5 1.53e-5 7.63e-6
do
  ./scripts/augusta/beaker/peteish1-wsd-anneal-launch.sh $NUM_NODES $LR $LOAD_STEP $ANNEAL_STEPS ${@}
done