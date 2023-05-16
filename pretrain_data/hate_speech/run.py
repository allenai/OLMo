"""

Runner for testing out Hate Speech & NSFW detection under new `Filter` framework.

From @aakankshan, @khyathic. @kylel refactored. 

"""

import argparse
import gzip
import json
import os
import time
from typing import Dict, List

from tqdm import tqdm

from pretrain_data.filters.src.ai2_llm_filters.data_types import (
    DocResult,
    Document,
    Span,
)
from pretrain_data.filters.src.ai2_llm_filters.filters import FastTextFilter


def read_jsonl_file(infile: str) -> List[Document]:
    docs = []
    with gzip.open(infile) as f_in:
        for line in f_in:
            doc = json.loads(line)
            if not doc["text"] or doc["text"].strip() == "":
                continue
            # TODO: common-crawl JSONs dont contain `version`
            doc = Document(
                source=doc["source"], version=doc.get("version"), id=doc["id"], text=doc["text"].lower().strip()
            )
            docs.append(doc)
    return docs


def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--infile", type=str)
    parse.add_argument("--outdir", type=str)
    parse.add_argument("--model_path", type=str)
    parse.add_argument("--model_name", type=str)
    parse.add_argument("--level", type=str)
    parse.add_argument("--sent_threshold", type=float)
    parse.add_argument("--batch", type=int)
    parse.add_argument("--head", type=int)
    parse.add_argument("--verbose", action="store_true")
    args = parse.parse_args()

    print(args)

    docs: List[Document] = read_jsonl_file(args.infile)
    
    if args.head:
        docs = docs[: args.head]
    level = args.level
    sent_threshold = args.sent_threshold if level == 'sent' else None
    batch_size = args.batch
    verbose = args.verbose if args.verbose else False
    model_path = args.model_path
    model_name = args.model_name

    filter = FastTextFilter(model_path=model_path, level=level, sent_threshold=sent_threshold)

    start_time = time.time()
    _infile = os.path.splitext(os.path.basename(args.infile))[0]
    outfile = os.path.join(args.outdir, f"{_infile}__model={model_name}__level={level}")
    if sent_threshold:
        outfile += f'__threshold={sent_threshold}'
    with open(outfile, mode="w") as f_out:
        for i, doc in enumerate(docs):
            if i % batch_size == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                start_time = time.time()
            doc_results = filter.predict(doc=doc)
            json.dump(doc_results.to_json(with_doc=verbose), f_out)
            f_out.write("\n")


if __name__ == "__main__":
    main()
