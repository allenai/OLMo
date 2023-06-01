import concurrent.futures
import pandas as pd
import os
import tqdm

key = "basic__whitespace_tokenizer_with_paragraphs_v1__document"


def func(s3_path, filep):
    path = os.path.join(s3_path, filep + ".json.gz")
    df = pd.read_json(path, lines=True, compression="gzip", storage_options={"config_kwargs": {"read_timeout": 600}})
    tokens = df["attributes"].apply(lambda x: x[key][0][2]).sum()
    return tokens, len(df)


if __name__ == "__main__":

    import sys
    import os

    version = sys.argv[2]
    s3_path = f"s3://ai2-llm/pretraining-data/sources/stack-dedup/{version}/attributes/basic"

    filep = sys.argv[1]

    outdir = sys.argv[2]
    path = f"{outdir}/{filep}.txt"
    if not os.path.exists(path): 
        x, y = func(s3_path, filep)
        with open(f"{outdir}/{filep}.txt", "w") as f:
            f.write(f"{filep}\t{x}\t{y}")
    print(f"{filep} done")
