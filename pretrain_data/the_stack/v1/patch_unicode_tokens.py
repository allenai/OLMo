import concurrent.futures
import logging
import os
import string
import sys
from typing import Dict, List

import pandas as pd
import tqdm

from pandarallel import pandarallel
from uniseg.wordbreak import words as unicode_tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_patch_unicode_tokens.log"),
    ],
)

pandarallel.initialize(progress_bar=False)

logger = logging.getLogger(__name__)

S3_location = "s3://ai2-llm/pretraining-data/sources/stack-dedup"
v0_atts = os.path.join(S3_location, "v0", "attributes", "file_stats")
v1_atts = os.path.join(S3_location, "v1", "attributes")

v0_docs = os.path.join(S3_location, "v0", "documents")
v1_docs = os.path.join(S3_location, "v1", "documents")


def count_tokens_unicode(text):
    # this is extremely slow
    count = sum(1 for word in unicode_tokenize(text) if not all(char in string.whitespace for char in word))
    return count


def copyright_tokens(merged_instance):
    return count_tokens_unicode(merged_instance["text_x"].replace(merged_instance["text_y"], ""))


def v2_unicode_tokens(filep):
    v0att = os.path.join(v0_atts, filep + ".tsv")
    v0adf = pd.read_csv(v0att, sep="\t")

    v1att = os.path.join(v1_atts, filep + ".tsv")
    v1adf = pd.read_csv(v1att, sep="\t")

    if "num_tokens_unicode" in v1adf.columns:
        return

    v0doc = os.path.join(v0_docs, filep + ".jsonl.gz")
    v0ddf = pd.read_json(v0doc, lines=True, compression="gzip")

    v1doc = os.path.join(v1_docs, filep + ".jsonl.gz")
    v1ddf = pd.read_json(v1doc, lines=True, compression="gzip")

    # try:
    #    v1adf = v1adf.drop(columns=["num_tokens_unicode"])
    # except KeyError:
    #    pass

    v1ma = pd.merge(v0adf, v1adf, on="id")
    v1md = pd.merge(v0ddf, v1ddf, on="id")

    to_remove = v1md.parallel_apply(copyright_tokens, axis=1)

    assert len(v1ma) == len(v1adf) == len(to_remove), (len(v1ma), len(v1adf), len(to_remove))

    v1adf["num_tokens_unicode"] = v1ma["num_tokens_unicode"] - to_remove
    error_ids = v1adf[v1adf["num_tokens_unicode"] <= 0]["id"]

    v1adf.loc[v1adf["id"].isin(error_ids), "num_tokens_unicode"] = v1md[v1md["id"].isin(error_ids)][
        "text_y"
    ].parallel_apply(count_tokens_unicode)
    v1adf.to_csv(v1att, sep="\t", index=False)
    logger.info("Done")


# df = pd.read_csv(os.path.join(S3_location, "raw_lang_infos.tsv"), sep="\t")


def _get_lang_list(lang_list_path: str) -> List[str]:
    with open(lang_list_path) as f:
        langs = f.readlines()

    langs = [lang.strip() for lang in langs]
    return langs


def run_single(filep):
    lang = filep.split("/")[0]
    langs = _get_lang_list("lang_list.txt")
    if lang not in langs:
        return
    v2_unicode_tokens(filep)


def run(lang_files: Dict):
    langs = _get_lang_list("lang_list.txt")
    with tqdm.tqdm(total=len(lang_files)) as pbar:
        for lang in lang_files:
            if lang not in langs:
                pbar.update(1)
                continue
            pbar.set_description(lang)
            with tqdm.tqdm(total=len(lang_files[lang])) as pbar2:
                with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                    attr_futures = []
                    for i, filep in enumerate(lang_files[lang]):
                        attr_futures.append(executor.submit(v2_unicode_tokens, filep))
                    for future in concurrent.futures.as_completed(attr_futures):
                        future.result()
                        pbar2.update(1)
            pbar.update(1)


if __name__ == "__main__":
    run_single(sys.argv[1])
