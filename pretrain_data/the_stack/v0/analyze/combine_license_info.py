import glob
import os
import sys
from ast import literal_eval
from typing import Dict

import pandas as pd


def combine_license_info(input_path: str, tokens_output_path: str, documents_output_path: str):
    all_licenses = []
    for path in glob.glob(os.path.join(input_path, "*", "*.tsv")):
        df = pd.read_csv(path, delimiter="\t", usecols=["url", "license_tokens", "license_counts", "lang"])
        all_licenses.append(df)
    ldf = pd.concat(all_licenses)
    ldf["license_tokens"] = ldf["license_tokens"].apply(literal_eval)
    ldf["license_counts"] = ldf["license_counts"].apply(literal_eval)
    license_counts: Dict[str, int] = {}
    license_tokens: Dict[str, int] = {}
    for i, row in ldf.iterrows():
        lang = row["lang"]
        tokens = row["license_tokens"]
        counts = row["license_counts"]
        if lang in license_tokens:
            lt = license_tokens[lang]
            lc = license_counts[lang]
        else:
            lt = {}
            lc = {}
        for lname, t in tokens.items():
            if lname in lt:
                lt[lname] += t
                lc[lname] += counts[lname]
            else:
                lt[lname] = t
                lc[lname] = counts[lname]
        license_tokens[lang] = lt
        license_counts[lang] = lc

        fldf = pd.DataFrame.from_dict(license_tokens)
        fldf = fldf.transpose()
        fldf = fldf.reset_index().rename(columns={"index": "lang"})
        fldf.to_csv(tokens_output_path, sep="\t", index=False)

        fldf = pd.DataFrame.from_dict(license_counts)
        fldf = fldf.transpose()
        fldf = fldf.reset_index().rename(columns={"index": "lang"})
        fldf.to_csv(documents_output_path, sep="\t", index=False)


if __name__ == "__main__":
    combine_license_info(sys.argv[1], sys.argv[2], sys.argv[3])
