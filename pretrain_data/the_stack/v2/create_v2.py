import argparse
import logging
import re
import string
import sys
from typing import Callable, Dict, List, Union

from pretrain_data.the_stack.create_utils import (
    _get_lang_list,
    should_exclude_filename,
    _get_documents_location,
    _get_attributes_location,
)
import pandas as pd
import s3fs

S3_FS = s3fs.S3FileSystem()

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

    if instance_attributes["num_tokens_unicode"] == 0:
        return False

    if instance_attributes["alpha_count"] / instance_attributes["num_tokens_unicode"] < 1.5:
        return False

    return True


def create_documents(old_version: str, new_version: str, filename: str, functions_to_apply: List[Callable]):
    # eg. filename: lang/data_0000
    old_url = f"{_get_documents_location(old_version)}/{filename}.jsonl.gz"
    old_attributes_url = f"{_get_attributes_location(old_version)}/{filename}.tsv"

    new_url = f"{_get_documents_location(new_version)}/{filename}.jsonl.gz"

    if not S3_FS.exists(new_url):
        logger.info(f"Creating document {new_url}")
        old_df = pd.read_json(old_url, lines=True, compression="gzip")
        old_adf = pd.read_csv(old_attributes_url, sep="\t")

        fdf = pd.merge(old_df, old_adf, on="id")
        for func in functions_to_apply:
            fdf = fdf[fdf.apply(func, axis=1)]

        fdf = fdf[old_df.columns]
        logger.info(f"{filename} - {len(old_df) - len(fdf)} / {len(old_df)} removed.")

        fdf.to_json(new_url, lines=True, compression="gzip", orient="records")

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
