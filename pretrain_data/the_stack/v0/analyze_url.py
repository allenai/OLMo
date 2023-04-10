import argparse
import string
import os
from ast import literal_eval
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tqdm
from uniseg.wordbreak import words as unicode_tokenize

S3_LOCATION = "s3://ai2-llm/pretraining-data/sources/stack-dedup/raw"


def count_tokens(text):
    count = sum(
        1 for word in unicode_tokenize(text) if not all(char in string.whitespace for char in word)
    )
    return count

def flatten(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


def process_url(url: str, output_file: str):

    name = url.split("/")[-1].replace(".parquet", ".jsonl.gz")
    lang = url.split("/")[-2]
    df = pd.read_json(S3_LOCATION + "/" + lang + "/" + name, lines=True, compression="gzip")
    num_documents = len(df)

    document_char_length = list(df["text"].apply(lambda x: len(x)).values)
    document_token_length = list(df["text"].apply(lambda x: count_tokens(x)).values)
    document_licenses = list(df["metadata"].apply(lambda x: x["max_stars_repo_licenses"]).values)
    total_tokens = sum(document_token_length)

    url_info = {
        "num_documents": num_documents,
        "document_char_length": document_char_length,
        "document_token_length": document_token_length,
        "document_licenses": document_licenses,
        "total_tokens": total_tokens,
    }
    url_df = pd.DataFrame.from_dict(url_info, orient="index")
    url_df = url_df.transpose().reset_index().rename(columns={"index":"url"})
    url_df["url"] = lang + "/" + name
    url_df["lang"] = lang
    url_df.to_csv(output_file, sep="\t", index=False)


def process_all_urls(urls: List[str], output_dir: str):

    with tqdm.tqdm(total=len(urls)) as pbar:
        for url in urls:
            name = url.split("/")[-1].replace(".parquet", ".tsv")
            lang = url.split("/")[-2]
            output_file = os.path.join(output_dir, lang, name)
            if os.path.exists(output_file):
                if os.path.getsize(output_path) > 0:
                    pbar.update(1)
                    continue
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            process_url(url, output_file)
            pbar.update(1)


def combine_urls(urls: List[str], output_file: str):
    lang_list = sorted(list(langs.keys()))

    if lang_to_analyze:
        lang_list = [lang_to_analyze]

    # converters = {"num_documents": literal_eval, "document_char_length": literal_eval, "document_token_length": literal_eval, "total_tokens": literal_eval}

    url_dfs = []
    with tqdm.tqdm(total=len(urls)) as pbar:
        for url in urls:
            name = url.split("/")[-1].replace(".parquet", ".tsv")
            lang = url.split("/")[-2]
            output_file = os.path.join(output_dir, lang, name)
            df = pd.read_csv(output_file, delimiter="\t")
            url_dfs.append(df)
            pbar.update(1)
    combined_df = pd.concat(url_dfs)
    combined_df.to_csv(output_file, sep="\t", index=False)

def run(url: str, output_dir: str):
    name = url.split("/")[-1].replace(".parquet", ".tsv")
    lang = url.split("/")[-2]
    output_file = os.path.join(args.output_dir, lang, name)
    if os.path.exists(output_file):
        if os.path.getsize(output_file) > 0:
            return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_url(url, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get statistics for a particular Stack file")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    run(args.url, args.output_dir)
