import argparse
from ast import literal_eval
from typing import Dict, List, Optional

import pandas as pd
import tqdm

S3_LOCATION = "s3://ai2-llm/pretraining-data/sources/stack-dedup/raw"


def run(langs: Dict[str, List[str]], output_file: str, lang_to_analyze: Optional[str] = None):
    lang_list = sorted(list(langs.keys()))

    if lang_to_analyze:
        lang_list = [lang_to_analyze]

    converters = {"doc_counts": literal_eval, "len_per_doc": literal_eval, "token_per_doc": literal_eval}
    try:
        ldf = pd.read_csv(
            output_file,
            delimiter="\t",
            converters=converters,
        )
        lang_info = ldf.set_index("lang").to_dict(orient="index")
    except FileNotFoundError:
        lang_info = {}

    print("Number of languages:", len(lang_list))

    with tqdm.tqdm(total=len(lang_list)) as outer_pbar:
        for i, lang in enumerate(lang_list):
            urls = langs[lang]
            print(lang, len(urls))
            outer_pbar.update(1)
            if lang in lang_info and len(lang_info[lang]["doc_counts"]) == len(urls):
                continue
            doc_counts = []
            token_per_doc = []
            len_per_doc = []
            with tqdm.tqdm(total=len(urls)) as pbar:
                for url in urls:
                    name = url.split("/")[-1].replace(".parquet", ".jsonl.gz")
                    df = pd.read_json(S3_LOCATION + "/" + lang + "/" + name, lines=True, compression="gzip")
                    doc_counts.append(len(df))
                    len_per_doc.append(list(df["text"].apply(lambda x: len(x)).values))
                    token_per_doc.append(list(df["text"].apply(lambda x: x.count(" ") + x.count("\n")).values))
                    pbar.update(1)

                    lang_info[lang] = {
                        "doc_counts": doc_counts,
                        "len_per_doc": len_per_doc,
                        "token_per_doc": token_per_doc,
                    }
                    ldf = pd.DataFrame.from_dict(lang_info, orient="index")
                    ldf.reset_index().rename(columns={"index": "lang"}).to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get some statistics about the The Stack (dedup)")
    parser.add_argument("--lang", type=str, required=False, default=None)
    parser.add_argument("--urls-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    langs = {}
    with open(args.urls_file) as f:
        urls = f.readlines()
        for url in urls:
            lang = url.split("/")[-2]
            if lang in langs:
                langs[lang].append(url.rstrip("\n"))
            else:
                langs[lang] = [url.rstrip("\n")]

    run(langs, args.output_file, args.lang)
