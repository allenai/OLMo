# This script downloads The Stack (dedup) from HuggingFace.
# GNU-parallel used for downloading in parallel.
# Reference: O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
URL_LIST_FILE=$1
if [[ -z "${HF_TOKEN}" ]]; then
	echo "Please set Huggingface API token as environment variable HF_TOKEN."
	echo "Also note that you may need to login to huggingface.co and accept the terms on the dataset page."
else
	echo $(wc -l $URL_LIST_FILE)
	cat $URL_LIST_FILE| parallel -j 10 --bar --max-args=1 ./download_url.sh {1} stack_tmp_dir
fi
