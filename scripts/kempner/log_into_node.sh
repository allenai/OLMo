#!/bin/bash

set -euxo pipefail

srun --interactive --pty --jobid=$1 bash