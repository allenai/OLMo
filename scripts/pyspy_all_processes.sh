#!/bin/bash

set -euo pipefail

export NODENAME=$(hostname -s)

# Redirect stdout and stderr so that we get a prefix with the node name
exec > >(trap "" INT TERM; sed -u "s/^/$NODENAME out: /")
exec 2> >(trap "" INT TERM; sed -u "s/^/$NODENAME err: /" >&2)

ps -x -o pid,comm | grep " python" | sed -r 's/^[ ]*([0-9]+) .*/\1/g' | while read i; do
  echo "Process $i:"
  py-spy dump --pid $i;
done
