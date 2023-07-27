#!/bin/bash

set -euo pipefail

ps -x -o pid,comm | grep " python" | sed -r 's/^[ ]?([0-9]+) .*/\1/g' | while read i; do
  echo "Process $i:"
  py-spy dump --pid $i;
done
