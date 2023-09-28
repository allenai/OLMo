""" a simple way to inspect lines that we don't decontaminate"""

import argparse
import json
import gzip
import random
import os
from collections import Counter
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
import datetime
import regex
import uniseg.wordbreak

def count_non_alpha_lines(input_file):
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    re_all_punctuation = regex.compile(
        r"^("
        r"[[:punct:]]|"
        r"\s|"
        r"["
        "\U0001F300-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\u2600-\u26FF\u2700-\u27BF"
        r"]+"
        r")+$",
        regex.UNICODE,
    )
    count = 0
    total = 0
    count_tokens = 0
    total_tokens = 0
    line_set = set()
    with gzip.open(input_file, 'rt') as f:
        for line in f:
            doc = json.loads(line.strip())
            for line in doc["text"].split("\n"):
                # n_tokens = len(tokenizer.tokenize(line))
                n_tokens = sum(1 for w in uniseg.wordbreak.words(line.strip()) if w.strip())
                total += 1
                total_tokens += n_tokens
                if n_tokens < 13:
                    continue
                if re_all_punctuation.match(line):
                    line_set.add(line)
                    count += 1
                    count_tokens += n_tokens
                
    return count, total, count_tokens, total_tokens, line_set

def main(args):


    with mp.Pool(mp.cpu_count()) as pool:
        counts_per_file = list(tqdm(pool.imap(count_non_alpha_lines, args.input_files), total=len(args.input_files), desc="reading input files"))
    count = 0
    total = 0
    count_tokens = 0
    total_tokens = 0
    line_set = set()
    for c, t, c_t, t_t, l_s in counts_per_file:
        count += c
        total += t
        count_tokens += c_t
        total_tokens += t_t
        line_set.update(l_s)


    print("punct_space_emoji lines", count)
    print("total lines", total)
    print("punct_space_emoji tokens", count_tokens)
    print("total tokens", total_tokens)
    print("unique lines", len(line_set))

    with open(args.output_file, 'w') as f:
        for line in line_set:
            f.write(line + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", help="input files jsonl.gz files to split")
    parser.add_argument("--output_file", type=str, help="output file for unique lines")
    args = parser.parse_args()

    main(args)