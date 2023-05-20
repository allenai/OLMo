import argparse
import datetime
import gzip
import json
import os
from uuid import uuid4

from datasets import load_dataset


class HFFormatter:
    @staticmethod
    def arc_challenge(args):
        dataset = load_dataset("ai2_arc", "ARC-Challenge")
        for item in dataset["test"]:
            yield {
                "id": item["id"],
                "text": item["question"],
            }


def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--dataset", type=str,
                       help="Dataset Name (must have corresponding HFFormatter method)")
    parse.add_argument("--out_dir", type=str, help="Output Directory")

    args = parse.parse_args()
    data = getattr(HFFormatter, args.dataset)(args)
    with gzip.open(os.path.join(args.out_dir, args.dataset + '.jsonl.gz'), 'wt') as fout:
        for doc in data:
            doc['id'] = str(uuid4()) if 'id' not in doc else doc['id']
            doc['added'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc['source'] = args.dataset if 'source' not in doc else doc['source']
            fout.write(json.dumps(doc) + '\n')


if __name__ == "__main__":
    main()
