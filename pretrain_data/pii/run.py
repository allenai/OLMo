"""

Run PII extraction with post processing rules over regexes

"""
import argparse
import gzip
import json
import os
import time
from typing import Dict, List

from pii_detector import WINDOW, DocResult, Document, PiiDetector, PiiSpan
from tqdm import tqdm


def read_jsonl_file(infile: str) -> List[Document]:
    docs = []
    with gzip.open(infile) as f_in:
        for line in f_in:
            doc = json.loads(line)
            if not doc["text"] or doc["text"].strip() == "":
                continue
            # common-crawl JSONs dont contain `version`
            doc = Document(
                source=doc["source"], version=doc.get("version"), id=doc["id"], text=doc["text"].lower().strip()
            )
            docs.append(doc)
    return docs


def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--infile", type=str)
    parse.add_argument("--method", type=str, help="regex or presidio")
    parse.add_argument("--postprocess", action="store_true")
    parse.add_argument("--batch", type=int)
    parse.add_argument("--window", type=int)
    parse.add_argument("--outdir", type=str)
    parse.add_argument("--verbose", action="store_true")
    parse.add_argument("--head", type=int)
    args = parse.parse_args()

    print(args)

    docs: List[Document] = read_jsonl_file(args.infile)
    if args.head:
        docs = docs[: args.head]

    pii_detector = PiiDetector()

    postprocess = args.postprocess
    method = args.method
    window = args.window if args.window else WINDOW
    verbose = args.verbose if args.verbose else False
    batch_size = args.batch

    start_time = time.time()
    _infile = os.path.splitext(os.path.basename(args.infile))[0]
    outfile = os.path.join(args.outdir, f"{_infile}__method={method}__postprocess={postprocess}__window={window}")
    with open(outfile, mode="w") as f_out:
        for i, doc in enumerate(docs):
            if i % batch_size == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                start_time = time.time()
            doc_results = pii_detector.predict(doc=doc, method=method, do_postprocess=postprocess, window=window)
            json.dump(doc_results.to_json(with_doc=verbose), f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()
