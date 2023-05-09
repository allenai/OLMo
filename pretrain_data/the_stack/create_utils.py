import logging
import sys
from typing import Callable, List

import pandas as pd
import s3fs

S3_FS = s3fs.S3FileSystem()
S3_LOCATION = "s3://ai2-llm/pretraining-data/sources/stack-dedup"

logger = logging.getLogger(__name__)


def _get_lang_list(lang_list_path: str) -> List[str]:
    with open(lang_list_path) as f:
        langs = f.readlines()

    langs = [lang.strip() for lang in langs]
    return langs


def _get_documents_location(version: str):
    base_location = f"{S3_LOCATION}/{version}"
    return f"{base_location}/documents"


def _get_attributes_location(version: str):
    base_location = f"{S3_LOCATION}/{version}"
    return f"{base_location}/attributes"


def create_documents(old_version: str, new_version: str, filename: str, functions_to_apply: List[Callable]):
    # eg. filename: lang/data_0000
    old_url = f"{_get_documents_location(old_version)}/{filename}.jsonl.gz"
    new_url = f"{_get_documents_location(new_version)}/{filename}.jsonl.gz"

    if not S3_FS.exists(new_url):
        logger.info(f"Creating document {new_url}")
        old_df = pd.read_json(old_url, lines=True, compression="gzip")
        old_df["new_text"] = old_df["text"]
        for func in functions_to_apply:
            old_df["new_text"] = old_df["new_text"].apply(func)

        changed = len(old_df[old_df["new_text"] != old_df["text"]])
        logger.info(f"{filename} - {changed} / {len(old_df)} were updated.")

        old_df["text"] = old_df["new_text"]
        old_df = old_df.drop(columns=["new_text"])

        old_df.to_json(new_url, lines=True, compression="gzip", orient="records")


def create_attributes(old_version: str, new_version: str, filename: str, functions_to_apply: List[Callable]):
    new_url = f"{_get_documents_location(new_version)}/{filename}.jsonl.gz"
    new_df = pd.read_json(new_url, lines=True, compression="gzip")

    new_attributes_url = f"{_get_attributes_location(new_version)}/{filename}.tsv"

    if not S3_FS.exists(new_attributes_url):
        logger.info(f"Creating attributes {new_attributes_url}")
        ndf = new_df
        for func in functions_to_apply:
            ndf = ndf.apply(func, axis=1)
        stat_keys = ["id"] + list(set(ndf.columns) - set(new_df.columns))
        ndf = ndf[stat_keys]

        ndf.to_csv(new_attributes_url, sep="\t", index=False)


def should_exclude_filename(filename: str, lang_list: List[str]) -> bool:
    lang = filename.split("/")[0]
    if lang not in lang_list:
        logger.warning(f"{filename} is excluded as it is not part of the selected languages.")
        return True
    return False
