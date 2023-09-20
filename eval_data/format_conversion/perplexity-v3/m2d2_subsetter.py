""" a subsetter just for m2d2"""

import argparse
import json
import gzip
import os
from transformers import AutoTokenizer
import multiprocessing as mp
from tqdm import tqdm
from uuid import uuid4
import datetime

def process_subdomain(subdomain_file, args):
    subdomain = subdomain_file.split('/')[-2]
    domain = subdomain_file.split('/')[-3]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # read subdomain_file line by line counting tokens until we reach args.tokens_per_subdomain
    token_count = 0
    text = ""
    with open(subdomain_file, 'rt') as f:
        for line in f:
            token_count += len(tokenizer.tokenize(line))
            if token_count > args.tokens_per_subdomain:
                break
            text += line

    if token_count == 0:
        return 0
    
    output_data = {
        "text": text,
        "id": str(uuid4()),
        "added": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": f'm2d2_{domain}_unsplit',
        "subdomain": subdomain
    }

    # ouput to args.output_dir/subdomain_file_name.jsonl.gz
    output_file = os.path.join(args.output_dir, f"{subdomain}.jsonl.gz")
    with gzip.open(output_file, 'wt') as f:
        f.write(json.dumps(output_data) + '\n')
    
    return token_count



def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare args for process_subdomain
    subdomain_args = [(subdomain_file, args) for subdomain_file in args.input_files]

    with mp.Pool(mp.cpu_count()) as pool:
        token_counts = list(tqdm(pool.starmap(process_subdomain, subdomain_args), total=len(subdomain_args)))

    for subdomain_file, token_count in zip(args.input_files, token_counts):
        print(f"{subdomain_file} token count: {token_count}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", help="input files jsonl.gz files to split")
    parser.add_argument("--output_dir", type=str, help="output directory to write the split files to")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="tokenizer to use for counting")
    parser.add_argument("--tokens_per_subdomain", type=int, default=100000, help="number of tokens to sample per subdomain")
    args = parser.parse_args()
    main(args)