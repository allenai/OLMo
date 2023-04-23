import glob
import os
import sys
from typing import Dict

import pandas as pd


def combine_stats(path: str, output_path: str):
    all_tokens = []
    for path in glob.glob(os.path.join(path, "*", "*.tsv")):
        df = pd.read_csv(
            path,
            delimiter="\t",
            usecols=["url", "min_tokens", "max_tokens", "total_tokens", "lang", "num_documents"],
        )
        all_tokens.append(df)

    tdf = pd.concat(all_tokens)
    tdf = tdf.reset_index().drop(columns=["index"])
    tdf = tdf.sort_values(by=["lang", "url"])
    lang_tokens: Dict[str, int] = {}
    min_tokens: Dict[str, int] = {}
    max_tokens: Dict[str, int] = {}
    num_documents: Dict[str, int] = {}
    for i, row in tdf.iterrows():
        lang = row["lang"]
        tt = row["total_tokens"]
        min_t = row["min_tokens"]
        max_t = row["max_tokens"]
        num_docs = row["num_documents"]
        if lang in lang_tokens:
            lang_tokens[lang] += tt
            min_tokens[lang] = min(min_tokens[lang], min_t)
            max_tokens[lang] = max(max_tokens[lang], max_t)
            num_documents[lang] += num_docs
        else:
            lang_tokens[lang] = tt
            min_tokens[lang] = min_t
            max_tokens[lang] = max_t
            num_documents[lang] = num_docs

    ltdf = pd.DataFrame.from_dict(lang_tokens, orient="index")
    ltdf = ltdf.reset_index().rename(columns={"index": "lang", 0: "total_tokens"})
    xtdf = pd.DataFrame.from_dict(min_tokens, orient="index")
    xtdf = xtdf.reset_index().rename(columns={"index": "lang", 0: "min_tokens"})
    ytdf = pd.DataFrame.from_dict(max_tokens, orient="index")
    ytdf = ytdf.reset_index().rename(columns={"index": "lang", 0: "max_tokens"})
    ztdf = pd.DataFrame.from_dict(num_documents, orient="index")
    ztdf = ztdf.reset_index().rename(columns={"index": "lang", 0: "num_documents"})

    atdf = pd.merge(ltdf, xtdf, on="lang")
    atdf = pd.merge(atdf, ytdf, on="lang")
    atdf = pd.merge(atdf, ztdf, on="lang")

    atdf.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    combine_stats(sys.argv[1], sys.argv[2])
