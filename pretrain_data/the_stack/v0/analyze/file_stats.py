# Reference: https://github.com/togethercomputer/RedPajama-Data/blob/main/data_prep/github/github_clean_dedup_local.py

import argparse
import logging
import os
import re
import string
import sys
from typing import Dict, Union

import pandas as pd
from uniseg.wordbreak import words as unicode_tokenize

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_file_stats.log"),
    ],
)

logger = logging.getLogger(__name__)


# Regex to strip repeated copyright comment blocks
CPAT = re.compile("copyright", re.IGNORECASE)
PAT = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")


def clean_copyright_comments(content: str):
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
        instance.update({
            "line_count": 0,
            "max_line_length": 0,
            "avg_line_length": 0,
            "alnum_prop": 0,
            "num_characters": 0,
            "num_tokens_unicode": 0,
            "num_tokens_whitespace": 0,
        })
        return instance

    num_characters = len(content)

    # get max line length
    max_length = max(line_lengths)

    # get average line length
    avg_length = num_characters / len(line_lengths)

    # get proportion of alphanumeric characters
    alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, content))
    alnum_prop = alnum_count / num_characters

    instance["line_count"] = len(line_lengths)
    instance["max_line_length"] = max_length
    instance["avg_line_length"] = avg_length
    instance["alnum_prop"] = alnum_prop
    instance["num_characters"] = num_characters
    instance["num_tokens_unicode"] = count_tokens_unicode(content) # nobody got time for that

    # whitespace
    instance["num_tokens_whitespace"] = len(content.split())

    return instance


def process_file(url: str, output_file: str, clean_copyright: bool, version: str):
    name = url.split("/")[-1] + ".jsonl.gz" #.replace(".parquet", ".jsonl.gz")
    lang = url.split("/")[-2]

    s3_location = f"s3://ai2-llm/pretraining-data/sources/stack-dedup/{version}/documents"

    s3_url = s3_location + "/" + lang + "/" + name

    logger.info(f"Processing url: {s3_url}")
    logger.info(f"Cleaning copyright comments for {lang}/{name}? {clean_copyright}")

    df = pd.read_json(s3_location + "/" + lang + "/" + name, lines=True, compression="gzip")
    ndf = df.apply(get_filecontent_stats, clean_copyright=clean_copyright, axis=1)

    stat_keys = ["id"] + list(set(ndf.columns) - set(df.columns))

    ndf = ndf[stat_keys]

    ndf.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Processed stats at: {output_file}")


def run(url: str, output_dir: str, clean_copyright: bool, version: str):
    json_url = url.split("/")[-1] + ".jsonl.gz" #.replace(".parquet", ".jsonl.gz")
    name = url.split("/")[-1] + ".tsv" #.replace(".parquet", ".tsv")
    lang = url.split("/")[-2]
    output_file = os.path.join(args.output_dir, lang, name)
    if os.path.exists(output_file):
        if os.path.getsize(output_file) > 0:
            logger.info(f"Statistics for {lang}/{json_url} already present at {output_file}.")
            return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_file(url, output_file, clean_copyright, version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get statistics for a particular Stack file")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--clean-copyright", default=False, action="store_true")
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    run(args.url, args.output_dir, args.clean_copyright, args.version)
