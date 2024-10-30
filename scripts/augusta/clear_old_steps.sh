#!/bin/bash

set -euxo pipefail

cd /mnt/localssd/runs
for run in *; do
	pushd $run
	for step in $(ls -1d step* | sort -t'p' -k2,2n | head -n -3); do
		rm -r $step
	done
	popd
done

