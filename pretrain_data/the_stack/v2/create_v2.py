import argparse
import logging
import os
import re
import string
import sys
from typing import Dict, Union, List, Callable

import pandas as pd
from uniseg.wordbreak import words as unicode_tokenize
import s3fs

S3_FS = s3fs.S3FileSystem()
S3_LOCATION = "s3://ai2-llm/pretraining-data/sources/stack-dedup"
V1_LOCATION = f"{S3_LOCATION}/v1"
V2_LOCATION = f"{S3_LOCATION}/v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_create_v2.log"),
    ],
)

logger = logging.getLogger(__name__)

def _get_lang_list(lang_list_path: str) -> List[str]:
    with open(lang_list_path) as f:
        langs = f.readlines()

    langs = [lang.strip() for lang in langs]
    return langs


def _get_documents_location(base_location: str):
    return f"{base_location}/documents"

def _get_attributes_location(base_location: str):
    return f"{base_location}/attributes"


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

def create_documents(filename: str, filter_functions: List[Callable]):
    # eg. filename: lang/data_0000
    v1_url = f"{_get_documents_location(V1_LOCATION)}/{filename}.jsonl.gz"
    v2_url = f"{_get_documents_location(V2_LOCATION)}/{filename}.jsonl.gz"

    v1_attributes_url = f"{_get_attributes_location(V1_LOCATION)}/{filename}.tsv"

    if not S3_FS.exists(v2_url):
        logger.info(f"Creating document {v2_url}")
        v1_df = pd.read_json(v1_url, lines=True, compression="gzip")
        v1_attribute_df = pd.read_csv(v1_attributes_url, sep="\t")

        v1 = pd.merge(v1_df, v1_attribute_df, on="id")
        for func in filter_functions:
            v1 = v1[v1.apply(func, axis=1)]

        removed = len(v1_df) - len(v1)
        logger.info(f"{filename} - {removed} / {len(v1_df)} were removed.")
        logger.info(f"{filename} - total remaining tokens: {v1['num_tokens_whitespace'].sum()}")

        v1 = v1[v1_df.columns]
        v1.to_json(v2_url, lines=True, compression="gzip", orient="records")


def create_attributes(filename: str, functions_to_apply: List[Callable]):
    v2_url = f"{_get_documents_location(V2_LOCATION)}/{filename}.jsonl.gz"
    v2_df = pd.read_json(v2_url, lines=True, compression="gzip")

    v2_attributes_url = f"{_get_attributes_location(V2_LOCATION)}/{filename}.tsv"

    if not S3_FS.exists(v1_attributes_url):
        logger.info(f"Creating attributes {v2_attributes_url}")
        ndf = v2_df
        for func in functions_to_apply:
            ndf = ndf.apply(func, axis=1)
        stat_keys = ["id"] + list(set(ndf.columns) - set(v2_df.columns))
        ndf = ndf[stat_keys]

        ndf.to_csv(v2_attributes_url, sep="\t", index=False)


def should_exclude_filename(filename: str, lang_list: List[str]) -> bool:
    lang = filename.split("/")[0]
    if lang not in lang_list:
        logger.warning(f"{filename} is excluded as it is not part of the selected languages.")
        return True
    return False


def process_file(filename: str):
    lang_list = _get_lang_list("lang_list.txt")
    if should_exclude_filename(filename, lang_list):
        return

    create_documents(filename, [filter_by_filecontent_stats])
    # create_attributes(filename, [get_filecontent_stats])
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create v2 files from corresponding v1 files.")
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    process_file(args.filename)

