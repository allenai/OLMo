import itertools
from collections import Counter

import numpy as np
from cached_path import cached_path
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from transformers import LlamaTokenizer

AI2_PATH = "s3://ai2-llm/tokenizer/model/v1.json"
LLAMA_PATH = "s3://ai2-s2-lucas/llama/tokenizer.model"


tokenizers = {
    "gpt": AutoTokenizer.from_pretrained("gpt2"),
    "ai2": PreTrainedTokenizerFast(tokenizer_file=str(cached_path(AI2_PATH))),
    "llama": LlamaTokenizer(vocab_file=str(cached_path(LLAMA_PATH))),
}


copa = load_dataset("super_glue", "copa", split="test")
lambada = load_dataset("EleutherAI/lambada_openai", "default", split="test")


def join_copa(row):
    return {"text": f'{row["premise"]} {row["choice1"]} {row["choice2"]} {row["label"]}'}


copa = copa.map(join_copa, remove_columns=copa.column_names)    # pyright: ignore

datasets = {"copa": copa, "lambada_openai": lambada}


for dn, dataset in datasets.items():
    for tn, tokenizer in tokenizers.items():
        print(f"{dn}/{tn}")

        tokens = tokenizer(dataset["text"], add_special_tokens=False).input_ids     # pyright: ignore
        lengths = [len(r) for r in tokens]
        counts = Counter(itertools.chain.from_iterable(tokens))
        avg = float(np.mean(lengths))
        std = float(np.std(lengths))

        print(f"avg: {avg:.2f} std: {std:.2f}")
        print("Most common tokens:")
        for k, v in counts.most_common(10):
            print(f"{repr(tokenizer.decode([k]))}: {v / sum(lengths):.2%}")

        print("\n")
