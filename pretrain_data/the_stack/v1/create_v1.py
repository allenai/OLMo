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
V0_LOCATION = f"{S3_LOCATION}/v0"
V1_LOCATION = f"{S3_LOCATION}/v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_create_v1.log"),
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


def clean_copyright_comments(content: str):
    # Regex to strip repeated copyright comment blocks
    CPAT = re.compile("copyright", re.IGNORECASE)
    PAT = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")

    r = PAT.search(content)
    if r:
        # found one, now see if it contains "copyright", if so strip it
        span = r.span()
        sub = content[span[0] : span[1]]
        if CPAT.search(sub):
            # cut it
            content = content[: span[0]] + content[span[1] :]

        return content

    lines = content.split("\n")
    skip = 0

    # Greedy replace any file that begins with comment block, most
    # are copyright headers
    for k in range(len(lines)):
        if lines[k].startswith("//") or lines[k].startswith("#") or lines[k].startswith("--") or not lines[k]:
            skip = skip + 1
        else:
            break

    if skip:
        # we skipped, consume it
        content = "\n".join(lines[skip:])

    return content


def count_tokens_unicode(text):
    # this is extremely slow
    count = sum(1 for word in unicode_tokenize(text) if not all(char in string.whitespace for char in word))
    return count


def get_filecontent_stats(instance, clean_copyright: bool = False) -> Dict[str, Union[int, str]]:
    # split content into lines and get line lengths
    content = instance["text"]
    if clean_copyright:
        content = clean_copyright_comments(content)

    line_lengths = list(map(len, content.splitlines()))

    if len(line_lengths) == 0:
        return {
            "line_count": 0,
            "max_line_length": 0,
            "avg_line_length": 0,
            "alnum_prop": 0,
            "num_characters": 0,
            "num_tokens_whitespace": 0,
            "num_alpha": 0
        }

    num_characters = len(content)

    # get max line length
    max_length = max(line_lengths)

    # get average line length
    avg_length = num_characters / len(line_lengths)

    num_tokens_whitespace = len(content.split())

    # get proportion of alphanumeric characters
    alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, content))
    alnum_prop = alnum_count / num_characters

    alpha_count = sum(map(lambda char: 1 if char.isalpha() else 0, content))
    # alpha_token_prop = alpha_count / num_tokens_whitespace

    instance["line_count"] = len(line_lengths)
    instance["max_line_length"] = max_length
    instance["avg_line_length"] = avg_length

    instance["alnum_count"] = alnum_count
    instance["alnum_prop"] = alnum_prop

    instance["alpha_count"] = alpha_count

    instance["num_characters"] = num_characters
    # instance["num_tokens_unicode"] = count_tokens_unicode(content) # nobody got time for that

    # whitespace
    instance["num_tokens_whitespace"] = num_tokens_whitespace

    return instance

def create_documents(filename: str, functions_to_apply: List[Callable]):
    # eg. filename: lang/data_0000
    v0_url = f"{_get_documents_location(V0_LOCATION)}/{filename}.jsonl.gz"
    v1_url = f"{_get_documents_location(V1_LOCATION)}/{filename}.jsonl.gz"

    #try:
    #    v1_df = pd.read_json(v1_url, lines=True, compression="gzip", chunksize=1)
    #except FileNotFoundError:
    if not S3_FS.exists(v1_url):
        logger.info(f"Creating document {v1_url}")
        v0_df = pd.read_json(v0_url, lines=True, compression="gzip")
        v0_df["new_text"] = v0_df["text"]
        for func in functions_to_apply:
            v0_df["new_text"] = v0_df["new_text"].apply(func)

        changed = len(v0_df[v0_df["new_text"] != v0_df["text"]])
        logger.info(f"{filename} - {changed} / {len(v0_df)} were updated.")

        v0_df["text"] = v0_df["new_text"]
        v0_df = v0_df.drop(columns=["new_text"])

        v0_df.to_json(v1_url, lines=True, compression="gzip", orient="records")


def create_attributes(filename: str, functions_to_apply: List[Callable]):
    v1_url = f"{_get_documents_location(V1_LOCATION)}/{filename}.jsonl.gz"
    v1_df = pd.read_json(v1_url, lines=True, compression="gzip")

    v1_attributes_url = f"{_get_attributes_location(V1_LOCATION)}/{filename}.tsv"

    #try:
    #    ndf = pd.read_csv(v1_attributes_url, sep="\t", chunksize=20)
    #except FileNotFoundError:
    if not S3_FS.exists(v1_attributes_url):
        logger.info(f"Creating attributes {v1_attributes_url}")
        ndf = v1_df
        for func in functions_to_apply:
            ndf = ndf.apply(func, axis=1)
        stat_keys = ["id"] + list(set(ndf.columns) - set(v1_df.columns))
        ndf = ndf[stat_keys]

        ndf.to_csv(v1_attributes_url, sep="\t", index=False)


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

    create_documents(filename, [clean_copyright_comments])
    create_attributes(filename, [get_filecontent_stats])
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create v1 files from corresponding v0 files.")
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    process_file(args.filename)

