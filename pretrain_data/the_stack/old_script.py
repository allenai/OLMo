"""
TODO:
    add better logging
    save to s3
"""
import argparse
import concurrent.futures
import gzip
import json
import os
from datetime import datetime
from multiprocessing import cpu_count
from typing import List

import fsspec
import pandas as pd
import tqdm


def _fs_auth(token: str):
    """
    Authenticate with huggingface token to allow dataset access.
    """
    headers = {"Authorization": f"Bearer {token}"}
    fs = fsspec.filesystem("https", headers=headers)
    return fs


def _read_parquet_with_token(file_path: str, token: str):
    """
    Authenticate to HF and read dataset parquet file into a pandas dataframe.
    """
    fs = _fs_auth(token)
    with fs.open(file_path, "rb") as f:
        df = pd.read_parquet(f)
    return df


def _convert_dataframe(
    df: pd.DataFrame,
    source: str,
    id_field: str,
    text_field: str,
    lang_field: str,
    timestamp: str,
    added: str,
):
    """
    Convert dataframe to the format we need.
    """
    for field in [text_field, id_field]:
        assert field in df.columns, f"'{field}' does not exist. Available fields: {df.columns}"

    df = df.rename(columns={text_field: "text", id_field: "id", lang_field: "lang"})
    remaining_columns = [col for col in df.columns if col not in ["id", "text", "lang"]]
    # df["lang"] = df["lang"].apply(lambda x: {x.lower().replace(" ", "-"): 1.0})
    df["lang"] = df["lang"].apply(lambda x: x.lower().replace(" ", "-"))
    df["metadata"] = df.filter(items=remaining_columns).to_dict(orient="records")
    df["timestamp"] = timestamp
    df["source"] = source
    df["added"] = added
    df = df.drop(columns=remaining_columns)
    return df


def _list_files(languages: List[str], lang_idx: int, total_files: int):
    """
    Get the list of parquet files for all languages in The Stack.
    """
    fs = _fs_auth(os.getenv("HF_TOKEN"))
    # total_files = 0
    lang_files = {}
    try:
        with tqdm.tqdm(total=len(languages)) as pbar:
            for idx, lang in enumerate(languages):
                if idx >= lang_idx:
                    num_data_file = 0
                    lang_files[lang] = []
                    pbar.set_description(f"Total files: {total_files}. Getting urls for {lang}")
                    # Huggingface parquet api does not allow listing parquet files for large
                    # datasets for some reason, so we construct urls by pattern matching.
                    # Also, we use version v1.1 instead of main, for reproducibility.
                    while True:
                        try:
                            url = (
                                "https://huggingface.co/datasets/bigcode/the-stack-dedup/"
                                f"resolve/v1.1/data/{lang}/data_{num_data_file:04d}.parquet"
                            )
                            if not fs.exists(url):
                                raise FileNotFoundError
                            total_files += 1
                            lang_files[lang].append(url)
                            num_data_file += 1
                        except FileNotFoundError:
                            break
                pbar.update(1)
    except KeyboardInterrupt:
        lang_files.pop(lang)  # remove last lang as it may be incomplete
        return lang_files
    return lang_files


def _save_url_as_jsonl(url: str, output_path: str):
    """
    Save parquet file as a jsonl.gz file in the expected format.
    """
    if os.path.exists(output_path):
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    token = os.getenv("HF_TOKEN")
    df = _read_parquet_with_token(url, token)
    df = _convert_dataframe(
        df,
        source="stack-dedup",
        id_field="hexsha",
        text_field="content",
        lang_field="lang",
        timestamp=datetime(
            2022, 12, 1
        ).isoformat(),  # v1.1 date. Source: https://huggingface.co/datasets/bigcode/the-stack-dedup/discussions/6
        added=datetime.utcnow().isoformat(),
    )
    js = df.to_json(orient="records", lines=True)
    with gzip.open(output_path, "wt") as f:
        f.write(js)


def _download_lang_files(list_of_urls: List[str], lang: str, num_workers: int, output_dir: str):
    """
    Download parquet files in parallel.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers,
        # thread_name_prefix=f"download_as_jsonl-",
    ) as executor:
        futures = {}
        with tqdm.tqdm(total=len(list_of_urls)) as pbar:  # , leave=False, desc=f"Processing {lang}") as pbar:
            for i, url in enumerate(list_of_urls):
                name = url.split("/")[-1].split(".parquet")[0]
                lang = url.split("/")[-2]
                file_path = os.path.join(output_dir, lang, f"{name}.jsonl.gz")
                future = executor.submit(_save_url_as_jsonl, url, file_path)
                future.add_done_callback(lambda p: pbar.update())
                futures[future] = f"{lang}/{name}.parquet"
            for future in concurrent.futures.as_completed(futures):
                pbar.set_description(f"Processing {futures[future]}")
                future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the-stack-dedup dataset to jsonl.gz files")
    parser.add_argument("--lang-list-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, required=False, default=cpu_count())

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(args.__dict__)

    with open(f"{args.output_dir}/download.config", "w+") as f:
        json.dump(args.__dict__, f, indent=4)

    with open(args.lang_list_file) as f:
        langs = json.load(f)
        langs = [k.lower().replace(" ", "-") for k in langs.keys()]

    lang_idx = 0
    lang_urls = {}
    try:
        with open(f"{args.output_dir}/lang_urls.json", "r+") as f:
            lang_urls.update(json.load(f))
        lang_idx = len(lang_urls)
        if lang_idx <= len(langs):
            raise FileNotFoundError
    except FileNotFoundError:
        urls = _list_files(langs, lang_idx, sum([len(v) for v in lang_urls.values()]))
        lang_urls.update(urls)
        with open(f"{args.output_dir}/lang_urls.json", "w+") as f:
            json.dump(lang_urls, f)

    # with tqdm.tqdm(total=len(lang_urls)) as pbar:
    #     for lang, urls in lang_urls.items():
    #         _download_lang_files(urls, lang, args.num_workers, args.output_dir)
    #         pbar.update()

    list_of_urls = []
    for _, urls in lang_urls.items():
        list_of_urls += urls
    print("Total files to process:", len(list_of_urls))
    _download_lang_files(list_of_urls, None, args.num_workers, args.output_dir)
