#!/usr/bin/env bash


# Flags
# -i: Input file containing the Wikipedia dump
# -o: Output directory where to store the extracted files

# Case statement to parse flags
while getopts i:o: flag
do
    case "${flag}" in
        i) input_file=${OPTARG};;
        o) output_dir=${OPTARG};;
    esac
done

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file does not exist"
    exit 1
fi

# Check if the output directory is specified
if [ -z "$output_dir" ]; then
    echo "Output directory not specified"
    exit 1
elif [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

set -ex
python -m wikiextractor.WikiExtractor \
    ${input_file} -o ${output_dir} \
    --processes 60 \
    --bytes 500M \
    --json \
    --namespaces "[[Article]]"
set +ex
