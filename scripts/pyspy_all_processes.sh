#!/bin/bash

set -euo pipefail

ps -x -o pid,comm | grep " python" | cut -d" " -f 2 | while read i; do
  py-spy dump --pid $i;
done
