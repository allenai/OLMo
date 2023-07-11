#!/usr/bin/env bash

# Author:   Luca Soldaini
# Email:    luca@soldaini.net

# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

names_file="${SCRIPT_DIR}/names.txt"
num_processes=8

process_file() {
    name=$1
    s3_prefix="s3://ai2-llm/pretraining-data/sources/stackexchange/raw/"
    url_prefix="https://archive.org/download/stackexchange/"

    # Check if file exists in s3 bucket
    if aws s3 ls ${s3_prefix}${name} > /dev/null 2>&1; then
        echo "File ${name} exists in s3 bucket."
    else
        echo "File ${name} does not exist in s3 bucket. Downloading and uploading..."
        wget -P "/tmp" ${url_prefix}${name}

        # Check if file downloaded successfully
        if [ -f "/tmp/${name}" ]; then
        echo "Successfully downloaded ${name}. Uploading to s3..."
        aws s3 cp "/tmp/${name}" ${s3_prefix}${name}
        rm ${name}   # remove the file from the machine
        else
        echo "Failed to download ${name}."
        fi
    fi
}

export -f process_file

# Use xargs to pass each line in names.txt to process_file function and run in parallel.
cat $names_file | xargs -I {} -P $num_processes bash -c "process_file {}"
