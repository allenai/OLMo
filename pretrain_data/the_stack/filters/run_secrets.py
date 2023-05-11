import argparse
import logging
import sys
from typing import Callable, List

import pandas as pd
import s3fs

from pretrain_data.the_stack.create_utils import (
    _get_attributes_location,
    _get_documents_location,
    _get_lang_list,
    should_exclude_filename,
)

from pretrain_data.the_stack.filters.secrets_filter import get_secrets, THE_STR

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

S3_FS = s3fs.S3FileSystem()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_run_secrets_parallel.log"),
    ],
)

logger = logging.getLogger(__name__)


def extract_code_secrets(instance) -> List[List]:
    secrets_spans: List[List] = []

    text = instance["text"]
    secrets = get_secrets(text)
    for _, secret in secrets:
        line_number = secret.line_number - 1
        span = secret.secret_value
        span_line = text.splitlines()[line_number]
        line_start = text.find(span_line)
        start = line_start + span_line.find(span)
        end = start + len(span)
        assert text[start:end] == span
        secret_type = secret.type.replace(" ", "_")
        secrets_spans.append([start, end, f"SECRET_{secret_type}", span])

    instance["secrets"] = secrets_spans
    return instance


def create_attributes(new_version: str, filename: str, functions_to_apply: List[Callable]):
    new_url = f"{_get_documents_location(new_version)}/{filename}.jsonl.gz"
    new_df = pd.read_json(new_url, lines=True, compression="gzip")

    lang = filename.split("/")[0]
    filep = filename.split("/")[1]
    new_attributes_url = f"{_get_attributes_location(new_version)}/{lang}/code_secrets/{filep}.jsonl"

    if not S3_FS.exists(new_attributes_url):
        logger.info(f"Creating attributes {new_attributes_url}")
        ndf = new_df
        for func in functions_to_apply:
            ndf = ndf.parallel_apply(func, axis=1)
        stat_keys = ["id"] + list(set(ndf.columns) - set(new_df.columns))
        ndf = ndf[stat_keys]

        ndf.to_json(new_attributes_url, lines=True, orient="records")

def process_file(new_version: str, lang_list_path: str, filename: str):
    lang_list = _get_lang_list(lang_list_path)
    if should_exclude_filename(filename, lang_list):
        return

    create_attributes(new_version, filename, [extract_code_secrets])
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new version files from corresponding old version files.")
    parser.add_argument("--new-version", type=str, required=False, default="v2")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--lang-list", type=str, required=False, default="lang_list.txt")
    args = parser.parse_args()

    process_file(args.new_version, args.lang_list, args.filename)
