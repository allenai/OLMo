import sys
from ast import literal_eval

import pandas as pd


def run(path: str):
    converters = {"document_token_length": literal_eval}
    df = pd.read_csv(path, sep="\t", converters=converters)
    df["min_tokens"] = [df.apply(lambda x: min(x.document_token_length), axis=1).values[0]]
    df["max_tokens"] = [df.apply(lambda x: max(x.document_token_length), axis=1).values[0]]
    df.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    run(sys.argv[1])
