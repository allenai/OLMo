"""
Use this to prepare a numpy memory-mapped language modeling dataset from raw *.json.gz
dataset files, such as those from c4. Each file is expected to be a gzipped JSON lines
file, which each JSON line has a field named "text" that is a string representing a single
document from the dataset.

To test out this script, run:

```bash
python scripts/prepare_memmap_dataset.py test_fixtures/*.json.gz -o /tmp/out.npy
```
"""

import concurrent.futures
import gzip
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import click
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from olmo import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def get_progress() -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        "files",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


def tokenize_file(tokenizer: Tokenizer, path: Path) -> Generator[List[int], None, None]:
    with gzip.open(path, "rt", encoding="UTF8") as f:
        for line in f:
            text = json.loads(line)["text"]
            yield tokenizer.encode(text, add_special_tokens=True)


def count_tokens(tokenizer_id: str, path: Path) -> Tuple[Path, int, int]:
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)
    num_tokens = 0
    num_docs = 0
    for token_ids in tokenize_file(tokenizer, path):
        num_tokens += len(token_ids)
        num_docs += 1
    return path, num_tokens, num_docs


def fill_memmap(tokenizer_id: str, path: Path, memmap_path: Path, num_tokens: int, offset: int, dtype: np.dtype):
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)
    memmap = np.memmap(memmap_path, mode="r+", dtype=dtype, offset=offset * dtype.itemsize, shape=(num_tokens,))
    index = 0
    for token_ids in tokenize_file(tokenizer, path):
        memmap[index : index + len(token_ids)] = token_ids
        index += len(token_ids)
    memmap.flush()


def validate_dataset(tokenizer_id: str, mmap_location: Path, dtype_str: str = "uint16", total_docs: int = None):
    log.info("Validating...")
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)
    dtype = np.dtype(dtype_str)
    memmap = np.memmap(mmap_location, mode="r", dtype=dtype)
    # Should have an EOS token for every document.
    # C4 files 0-100 and GPT-Neo-20b tokenizer does not have this property
    # total_docs = 35631728 (memmap == tokenizer.eos_token_id).sum() = 36093844
    #if total_docs is not None:
    #    assert (memmap == tokenizer.eos_token_id).sum() == total_docs
    assert memmap[-1] == tokenizer.eos_token_id
    # Make sure all entries have been filled with actual token IDs.
    assert (memmap < tokenizer.vocab_size).all()
    print(tokenizer.decode(memmap[:100]))
    log.info("All good!")


@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Specify the output path.",
    prompt="Output file",
)
@click.option(
    "--tokenizer", "tokenizer_id", type=str, help="Name of path of a pretrained tokenizer", default="gpt2"
)
@click.option("--dtype", "dtype_str", default="uint16")
@click.option("--validate/--no-validate", default=False)
@click.option("-j", "--workers", "max_workers", type=int, default=None, help="Defaults to number of CPUs")
def main(
    src: Tuple[Path],
    output: Path,
    tokenizer_id: str,
    dtype_str: str,
    validate: bool,
    max_workers: Optional[int] = None,
):
    dtype = np.dtype(dtype_str)
    dtype_max = np.iinfo(dtype).max

    # Tokenize all documents to determine how many tokens are in each file.
    src_to_num_tokens: Dict[Path, int] = defaultdict(int)
    total_docs = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(max_workers or os.cpu_count() or 1, len(src))
    ) as executor:
        futures = []
        for path in src:
            future = executor.submit(count_tokens, tokenizer_id, path)
            futures.append(future)
        with get_progress() as progress:
            for future in progress.track(
                concurrent.futures.as_completed(futures), description="Counting tokens...", total=len(futures)
            ):
                path, num_tokens, num_docs = future.result()
                src_to_num_tokens[path] = num_tokens
                total_docs += num_docs

    total_tokens = sum(src_to_num_tokens.values())
    log.info(f"Counted {total_tokens:,d} tokens over {total_docs:,d} documents")

    # Initialize memmap file.
    memmap = np.memmap(output, mode="w+", dtype=dtype, shape=(total_tokens,))
    if validate:
        # Fill with max value so that we can check later that all values in the array
        # have been populated with actual token IDs.
        memmap[:] = dtype_max
    memmap.flush()
    del memmap

    # Now tokenizer all documents again and populate the memmap array.
    # We do this in parallel.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(max_workers or os.cpu_count() or 1, len(src))
    ) as executor:
        futures = []
        offset = 0
        for path in sorted(src):
            future = executor.submit(
                fill_memmap, tokenizer_id, path, output, src_to_num_tokens[path], offset, dtype
            )
            futures.append(future)
            offset += src_to_num_tokens[path]
        with get_progress() as progress:
            for future in progress.track(
                concurrent.futures.as_completed(futures), description="Filling memmap array...", total=len(futures)
            ):
                future.result()

    log.info(f"Done! File written to {output}")

    if validate:
        validate_dataset(tokenizer_id, output, dtype_str, total_docs)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    prepare_cli_environment()
    main()
