import argparse
import logging
import re
import string
import sys
from typing import Dict, Union

from pretrain_data.the_stack.create_utils import (
    _get_lang_list,
    create_documents,
    should_exclude_filename,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_create_v2.log"),
    ],
)

logger = logging.getLogger(__name__)

def filter_by_filecontent_stats(instance_attributes) -> bool:

    if instance_attributes["max_line_length"] > 1000:
        return False

    if instance_attributes["avg_line_length"] > 100:
        return False

    if instance_attributes["alnum_prop"] < 0.25:
        return False

    if instance_attributes["num_tokens_whitespace"] < 50:
        return False

    if instance_attributes["alpha_count"] / instance_attributes["num_tokens_whitespace"] < 1.5:
        return False

    return True


def create_attributes(old_version: str, new_version: str, filename: str):
    v2_url = f"{_get_documents_location(new_version)}/{filename}.jsonl.gz"
    v2_df = pd.read_json(v2_url, lines=True, compression="gzip")

    v2_attributes_url = f"{_get_attributes_location(new_version)}/{filename}.tsv"

    v1_attributes_url = f"{_get_attributes_location(old_version)}/{filename}.tsv"
    v1_adf = pd.read_csv(v1_attributes_url, sep="\t")

    if not S3_FS.exists(v2_attributes_url):
        logger.info(f"Creating attributes {v2_attributes_url}")
        ndf = v1_adf[v1_adf["id"].isin(v2_df["id"])]
        ndf.to_csv(v2_attributes_url, sep="\t", index=False)


def process_file(old_version: str, new_version: str, lang_list_path: str, filename: str):
    lang_list = _get_lang_list(lang_list_path)
    if should_exclude_filename(filename, lang_list):
        return

    create_documents(old_version, new_version, filename, [filter_by_filecontent_stats])
    create_attributes(old_version, new_version, filename)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new version files from corresponding old version files.")
    parser.add_argument("--old-version", type=str, required=False, default="v1")
    parser.add_argument("--new-version", type=str, required=False, default="v2")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--lang-list", type=str, required=False, default="lang_list.txt")
    args = parser.parse_args()

    process_file(args.old_version, args.new_version, args.lang_list, args.filename)
