import concurrent.futures
import os
from ast import literal_eval
from typing import Dict

import pandas as pd
import tqdm

S3_location = "s3://ai2-llm/pretraining-data/sources/stack-dedup"
v0_atts = os.path.join(S3_location, "v0", "attributes", "file_stats")


def patch(filep, lang_tokens_i):
    v0att = os.path.join(v0_atts, filep + ".tsv")
    v0adf = pd.read_csv(v0att, sep="\t")

    assert len(v0adf) == len(lang_tokens_i), f"{filep} token length error"
    v0adf["num_tokens_unicode"] = lang_tokens_i

    v0adf.to_csv(v0att, sep="\t", index=False)


# df = pd.read_csv(os.path.join(S3_location, "raw_lang_infos.tsv"), sep="\t")


def run(lang_files: Dict):
    with tqdm.tqdm(total=len(lang_files)) as pbar:
        for lang in lang_files:
            pbar.set_description(lang)

            with tqdm.tqdm(total=len(lang_files[lang])) as pbar2:
                with concurrent.futures.ThreadPoolExecutor(
                    thread_name_prefix=f"{lang}", max_workers=20
                ) as executor:
                    attr_futures = []
                    for i, filep in enumerate(lang_files[lang]):
                        ldf = pd.read_csv(os.path.join(S3_location, "v0/statistics", filep + ".tsv"), sep="\t")
                        lang_tokens_i = literal_eval(ldf["document_token_length"].values[0])
                        attr_futures.append(executor.submit(patch, filep, lang_tokens_i))
                    for future in concurrent.futures.as_completed(attr_futures):
                        future.result()
                        pbar2.update(1)
            pbar.update(1)


if __name__ == "__main__":
    import json

    with open("../lang_files.json") as f:
        lang_files = json.load(f)

    run(lang_files)
