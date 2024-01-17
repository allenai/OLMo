"""
Script for preparing the Tulu V2 data for fine-tuning an OLMo model.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)

    dataset = ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train")

    log.info("Tokenizing dataset...")
    preprocessed = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
        batched=False,
        remove_columns=["dataset", "id", "messages"],
        num_proc=opts.num_proc,  # type: ignore
    )

    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(preprocessed):
        assert len(ex["input_ids"]) == opts.seq_len  # type: ignore
        total_tokens += len(ex["input_ids"])  # type: ignore
    log.info(f"{total_tokens:,d}")

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"), dtype=np.uint16, mode="w+", shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
    )
    offset = 0
    for ex in track(preprocessed):
        ex_len = len(ex["input_ids"])  # type: ignore
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    parts = []
    for msg in example["messages"]:
        parts.append(f"<|{msg['role']}|>")
        parts.append(msg["content"])

    prompt = "\n".join(parts[:-1]) + "\n"
    completion = parts[-1]

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=True)

    input_ids = (prompt_ids + completion_ids)[:max_seq_len]
    label_mask = ([False] * len(prompt_ids) + [True] * len(completion_ids))[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Tulu V2 dataset")
    parser.add_argument("output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default="tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=2048)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=0)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)
