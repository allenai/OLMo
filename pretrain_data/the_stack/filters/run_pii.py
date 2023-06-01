import argparse
import logging
import sys
from typing import Callable, List

import pandas as pd
import s3fs
from pandarallel import pandarallel

from pretrain_data.filters.src.ai2_llm_filters.data_types import Document
from pretrain_data.filters.src.ai2_llm_filters.filters import PiiFilter
from pretrain_data.the_stack.create_utils import (
    _get_attributes_location,
    _get_documents_location,
    _get_lang_list,
    should_exclude_filename,
)

pandarallel.initialize(progress_bar=False)

S3_FS = s3fs.S3FileSystem()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_run_pii.log"),
    ],
)

logger = logging.getLogger(__name__)

PII_FILTER = PiiFilter(method="regex", postprocess=True, window=PiiFilter.WINDOW)


def extract_pii(instance):
    doc = Document(
        source=instance["source"],
        version=instance.get("version"),
        id=instance["id"],
        text=instance["text"].lower().strip(),
    )
    try:
        doc_results = PII_FILTER.predict(doc=doc)
        results_json = doc_results.to_json(with_doc=False)

        instance["score"] = results_json["score"]
        instance["spans"] = results_json["spans"]
    except ZeroDivisionError:
        instance["score"] = -1.0
        instance["spans"] = []
    return instance


def create_attributes(new_version: str, filename: str, functions_to_apply: List[Callable]):
    #new_url = f"{_get_documents_location(new_version)}/{filename}.jsonl.gz"
    #new_df = pd.read_json(new_url, lines=True, compression="gzip")

    lang = filename.split("/")[0]
    filep = filename.split("/")[1]
    new_attributes_url = f"{_get_attributes_location(new_version)}/{lang}/pii/{filep}.jsonl__method=regex__postprocess=True__window=100"

    if not S3_FS.exists(new_attributes_url):
        new_url = f"{_get_documents_location(new_version)}/{filename}.jsonl.gz"
        new_df = pd.read_json(new_url, lines=True, compression="gzip")
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

    create_attributes(new_version, filename, [extract_pii])
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new version files from corresponding old version files.")
    parser.add_argument("--new-version", type=str, required=False, default="v2")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--lang-list", type=str, required=False, default="lang_list.txt")
    args = parser.parse_args()

    process_file(args.new_version, args.lang_list, args.filename)
