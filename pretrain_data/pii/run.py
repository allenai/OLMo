""" Run CC shard PII extraction with post processing rules over regexes"""
import argparse
import gzip
import json
import time
from typing import Dict, List

from pii_detector import DocResult, Document, PiiDetector, PiiSpan
from tqdm import tqdm

start_time = time.time()


def read_jsonl_file(infile: str) -> List[Document]:
    docs = []
    with gzip.open(infile) as f_in:
        for line in f_in:
            doc = json.loads(line)
            if not doc["text"] or doc["text"].strip() == "":
                continue
            doc = Document(
                source=doc["source"], version=doc["version"], id=doc["id"], text=doc["text"].lower().strip()
            )
            docs.append(doc)
    return docs


def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--infile", type=str)
    parse.add_argument("--method", type=str, help="regex or presidio")
    parse.add_argument("--postprocess", action="store_true")
    parse.add_argument("--window", type=int)
    parse.add_argument("--outfile", type=str)
    parse.add_argument("--verbose", action="store_true")
    args = parse.parse_args()

    docs: List[Document] = read_jsonl_file(args.infile)

    pii_detector = PiiDetector()

    postprocess = args.postprocess
    method = args.method
    window = args.window if args.window else None
    verbose = args.verbose if args.verbose else False

    with open(args.outfile, mode="w") as f_out:
        for doc in tqdm(docs):
            doc_results = pii_detector.predict(doc=doc, method=method, do_postprocess=postprocess, window=window)
            json.dump(doc_results.to_json(with_doc=verbose), f_out)
            f_out.write("\n")

    print(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
