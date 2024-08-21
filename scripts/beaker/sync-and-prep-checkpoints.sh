#!/usr/bin/env bash

set -exuo pipefail

# get args
FROM_PATH=$1
shift
TO_PATH=$1
shift
SYNC_KWARGS=$*

aws s3 sync --dryrun "$FROM_PATH" "$TO_PATH" $SYNC_KWARGS
