#!/bin/bash

# path to current script
SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Set the name of the file with the list of urls
URL_FILE="${SCRIPT_PATH}/urls.txt"

# Set the maximum number of parallel downloads
MAX_PARALLEL=64

# save it here
LOG_FILE="downloaded_files.log"

PARALLEL_SCRIPT=$(cat <<END
# python script starts here

print('Hello, World!')

# script ends here
END
)


# Use GNU Parallel to download and upload files in parallel
cat "$URL_FILE" | parallel -j "$MAX_PARALLEL" --gnu '
    url=$(echo "{}" | python -c "import sys, string; x = sys.stdin.read(); print(x if x[0] != string.punctuation[6] else x[1:-2])")

    prefix="https://data.together.xyz/redpajama-data-1T/v1.0.0/"

    # Remove prefix from url
    remote_filename="${url#$prefix}"

    # for local filename, replace all "/" with "_"
    filename="${remote_filename//\//_}"

    # Check if filename exists on S3
    if aws s3 ls "s3://ai2-llm/pretraining-data/sources/redpajama/raw/metadata/${remote_filename}.done" > /dev/null 2>&1; then
        echo "File ${filename} already exists on S3"
    else
        echo "Downloading ${url}"

        curl -L "${url}" > "/tmp/${filename}" && echo "${filename}" >&1

        # check if file is zst compressed or not
        if file "/tmp/${filename}" | grep -q "Zstandard"; then
            aws s3 cp "/tmp/${filename}" "s3://ai2-llm/pretraining-data/sources/redpajama/raw/data/${remote_filename}"

            # create note that is done
            echo "Done with ${filename}" > /tmp/${filename}.done

            # Upload the done file to S3
            aws s3 cp "/tmp/${filename}.done" "s3://ai2-llm/pretraining-data/sources/redpajama/raw/metadata/${remote_filename}.done"
        else
            # Gzip the file
            gzip "/tmp/${filename}"

            # Upload the file to S3
            aws s3 cp "/tmp/${filename}.gz" "s3://ai2-llm/pretraining-data/sources/redpajama/raw/data/${remote_filename}.gz"

            echo "Done with ${filename}" > /tmp/${filename}.done

            # Upload the done file to S3
            aws s3 cp "/tmp/${filename}.done" "s3://ai2-llm/pretraining-data/sources/redpajama/raw/metadata/${remote_filename}.done"
        fi

        # Remove all files
        rm /tmp/${filename}*
    fi
'
