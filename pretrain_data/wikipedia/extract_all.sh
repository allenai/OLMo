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


# Case statement to parse flags
while getopts i:o: flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        o) output_dir=${OPTARG};;
    esac
done


# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Input directory does not exist"
    exit 1
fi

# Check if the output directory is specified
if [ -z "$output_dir" ]; then
    echo "Output directory not specified"
    exit 1
elif [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# remove trailing slash from input_dir and output_dir
input_dir=${input_dir%/}
output_dir=${output_dir%/}

# loop over all files in the input directory
for file in $input_dir/*; do
    # get the filename
    filename=$(basename -- "$file")

    # remove all extensions
    filename="${filename%.*.*}"

    # run extaction for each file
    bash ${SCRIPT_DIR}/extract.sh -i ${file} -o ${output_dir}/${filename}
done
