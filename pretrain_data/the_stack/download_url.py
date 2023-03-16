import gzip
import os
import sys
from datetime import datetime

import fsspec
import pandas as pd


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


def download(url: str, output_dir: str):
    name = url.split("/")[-1].split(".parquet")[0]
    lang = url.split("/")[-2]
    file_path = os.path.join(output_dir, lang, f"{name}.jsonl.gz")
    _save_url_as_jsonl(url, file_path)


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
