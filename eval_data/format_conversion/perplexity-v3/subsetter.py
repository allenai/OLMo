""" a much simpler subsetter"""

import argparse
import json
import gzip
import random
import os
from collections import Counter
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp

def split_by_total_docs(data_sources, args):
    docs_per_split_per_file = [len(d) // len(args.split_names) for d in data_sources]

    # split the data
    splits = {}
    for i, split_name in enumerate(args.split_names):
        for data, docs_per_split in zip(data_sources, docs_per_split_per_file):
            splits[split_name] = splits.get(split_name, []) + data[i*docs_per_split:(i+1)*docs_per_split]
        # report subdomains per split
        subdomain_counts = Counter([doc['subdomain'] for doc in splits[split_name]])
        print(f"subdomains in {split_name}:")
        for k,v in subdomain_counts.items():
            print(f"{k} {v}")
    return splits

def split_by_tokens(data_sources, args):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    target_tokens_per_source = args.split_token_count_target // len(data_sources)
    print(f"target tokens per source: {target_tokens_per_source}")

    # split the data
    splits = {}
    for i, split_name in enumerate(args.split_names):
        splits[split_name] = []
        for data in data_sources:
            token_count = 0
            while len(data) > 0:
                doc = data.pop()
                token_count += len(tokenizer.tokenize(doc['text']))
                splits[split_name].append(doc)
                if token_count > target_tokens_per_source:
                    break
            print(f"{split_name} source {i} token count: {token_count}")
    return splits

def read_input_file(input_file):
    with gzip.open(input_file, 'rt') as f:
        return [json.loads(line.strip()) for line in f]

def main(args):

    # set the random seed
    random.seed(args.seed)

    with mp.Pool(mp.cpu_count()) as pool:
        data_sources = list(tqdm(pool.imap(read_input_file, args.input_files), total=len(args.input_files), desc="reading input files"))

    
    # remove len 0 data sources
    data_sources = [data for data in data_sources if len(data) > 0]

    if not args.sample_evenly_by_file:
        data_sources = [[doc for data in data_sources for doc in data]]
    # shuffle the data
    for data in data_sources:
        random.shuffle(data)
    
    if args.split_token_count_target:
        splits = split_by_tokens(data_sources, args)
    else:
        splits = split_by_total_docs(data_sources, args)

    
    # write the splits to shards of at most args.shard_size_target bytes
    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_data in splits.items():
        os.makedirs(os.path.join(args.output_dir, split_name), exist_ok=True)
        shard_num = 0
        shard_size = 0
        with gzip.open(os.path.join(args.output_dir, split_name, f"{split_name}-{shard_num:08}.jsonl.gz"), 'wt') as f:
            for doc in split_data:
                doc_str = json.dumps(doc)
                doc_size = len(doc_str.encode('utf-8'))
                if shard_size + doc_size > args.shard_size_target:
                    # print(f"shard {shard_num} size: {shard_size}")
                    shard_num += 1
                    shard_size = 0
                    f.close()
                    # pad the shard number with zeros so that the files sort nicely
                    f = gzip.open(os.path.join(args.output_dir, split_name, f"{split_name}-{shard_num:08}.jsonl.gz"), 'wt')
                f.write(doc_str + "\n")
                shard_size += doc_size
        f.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", help="input files jsonl.gz files to split")
    parser.add_argument("--output_dir", type=str, help="output directory to write the split files to")
    parser.add_argument("--split_names", type=str, nargs="+", default=['val','test'], help="names of the splits to create")
    parser.add_argument("--split_token_count_target", type=int, help="target number of tokens per split, defaults to split by docs")
    parser.add_argument("--shard_size_target", type=int, default=1000000, help="target number of bytes per shard")
    parser.add_argument("--sample_evenly_by_file", action="store_true", help="sample evenly from each input file")
    parser.add_argument("--tokenizer", type=str, help="tokenizer to use for counting")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args)