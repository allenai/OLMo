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
import contextlib
import functools
import gzip
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import click
import msgspec
import numpy as np
import tqdm
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from smashed.utils.io_utils import (
    MultiPath,
    decompress_stream,
    open_file_for_write,
    recursively_list_files,
    stream_file_for_read,
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


class InputDocumentSpec(msgspec.Struct):
    # almost 5x faster than built-in json decoding in my tests;
    # can work with approximate spec (i.e., ignore missing fields)
    text: str


def tokenize_file(tokenizer: Tokenizer, path: str, batch_size: int = 1_000) -> Generator[List[int], None, None]:
    decoder = msgspec.json.Decoder(InputDocumentSpec)

    with ExitStack() as stack:
        input_file = stack.enter_context(stream_file_for_read(path, mode="rb"))
        input_stream = stack.enter_context(decompress_stream(input_file, mode="rt"))

        for line in input_stream:
            row = decoder.decode(line)
            yield tokenizer.encode(row.text, add_special_tokens=True)


@contextmanager
def make_mmap_file(
    path: str,
    dtype: np.dtype,
    max_tokens: int = 2 * 1024 * 1024 * 1024,  # 2B tokens * 2 bytes per token (uint16) = 4GB
) -> Iterator[np.memmap]:
    """Make a memory-mapped file."""
    parsed_path = MultiPath.parse(path)
    if parsed_path.is_local:
        local_memmap_path = parsed_path.as_path

        # make sure the directory exists
        local_memmap_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        with NamedTemporaryFile(delete=False) as f:
            # if the destination for the memmap is not local, we need to write to a temporary file first
            local_memmap_path = Path(f.name)

    memmap = np.memmap(local_memmap_path, mode="w+", dtype=dtype, shape=(max_tokens,))

    log.info(f"Created memmap file at {local_memmap_path} of size {memmap.nbytes:,} bytes")
    yield memmap

    # write the memmap to the destination
    memmap.flush()
    if not parsed_path.is_local:
        with stream_file_for_read(local_memmap_path, "rb") as f, open_file_for_write(parsed_path, mode="wb") as g:
            g.write(f.read())
        log.info(f"Written memmap file to {parsed_path.as_str}")

        # delete the temporary file
        os.remove(local_memmap_path)


def fill_memmap(
    tokenizer_id: str,
    path: str,
    memmap_path: str,
    dtype: np.dtype,
    max_tokens: int = 2 * 1024 * 1024 * 1024,  # 2B tokens * 2 bytes per token (uint16) = 4GB
):
    file_index = 0
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)

    memmap: Optional[np.memmap] = None
    tokens_index = 0

    with ExitStack() as stack:
        for token_ids in tqdm.tqdm(tokenize_file(tokenizer=tokenizer, path=path)):
            if memmap is None or tokens_index + len(token_ids) >= max_tokens:
                stack.pop_all().close()
                current_memmap_path = f"{memmap_path}_{file_index:05d}.npy"
                memmap = stack.enter_context(
                    make_mmap_file(path=current_memmap_path, dtype=dtype, max_tokens=max_tokens)
                )
                file_index += 1

            memmap[tokens_index : tokens_index + len(token_ids)] = token_ids
            tokens_index += len(token_ids)

        stack.pop_all().close()


def make_source_and_target(src: Tuple[str, ...], output: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    exploded_src: List[str] = []
    exploded_dst: List[str] = []

    parsed_output = MultiPath.parse(output)
    for prefix in src:
        parsed_prefix = MultiPath.parse(prefix)
        for path in recursively_list_files(parsed_prefix):
            exploded_src.append(path)
            exploded_dst.append((parsed_output / MultiPath.parse(path) - parsed_prefix).as_str.replace(".", "_"))

    return tuple(sorted(exploded_src)), tuple(sorted(exploded_dst))


@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=str,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Specify the output path.",
    prompt="Output file",
)
@click.option(
    "--tokenizer", "tokenizer_id", type=str, help="Name of path of a pretrained tokenizer", default="gpt2"
)
@click.option("--dtype", "dtype_str", default="uint16")
@click.option("--validate/--no-validate", default=False)
@click.option(
    "--max-tokens",
    default=2 * 1024 * 1024 * 1024,
    type=int,
    help="Maximum number of tokens to store in a single memmap file (default: 2B tokens or 4GB)",
)
@click.option("--debug/--no-debug", default=False, help="Enable debug (single process mode)")
@click.option("-j", "--workers", "max_workers", type=int, default=None, help="Defaults to number of CPUs")
def main(
    src: Tuple[str, ...],
    output: str,
    tokenizer_id: str,
    dtype_str: str,
    validate: bool,
    max_tokens: int,
    debug: bool,
    max_workers: Optional[int] = None,
):
    dtype = np.dtype(dtype_str)
    src, dst = make_source_and_target(src=src, output=output)

    # creating a partial here with all the arguments we need to pass to fill_memmap except for the paths
    # so that we don't make mistakes between debug and non-debug mode
    fill_memmap_fn = functools.partial(fill_memmap, tokenizer_id=tokenizer_id, dtype=dtype, max_tokens=max_tokens)

    if debug:
        log.info("Running in debug mode. Only one process will be used.")
        for src_path, dst_path in zip(src, dst):
            fill_memmap_fn(path=src_path, memmap_path=dst_path)
        return

    # Now tokenizer all documents again and populate the memmap array. We do this in parallel.
    workers_cnt = min(max_workers or os.cpu_count() or 1, len(src))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_cnt) as executor:
        futures: List[Future[None]] = []
        for src_path, dst_path in zip(src, dst):
            future = executor.submit(fill_memmap_fn, path=src_path, memmap_path=dst_path)
            futures.append(future)
        with get_progress() as progress:
            for future in progress.track(
                concurrent.futures.as_completed(futures),
                description="Filling memmap arrays...",
                total=len(futures),
            ):
                future.result()

    log.info(f"Done! File written to {output}")

    # if validate:
    #     log.info("Validating...")
    #     tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)
    #     memmap = np.memmap(output, mode="r", dtype=dtype, shape=(total_tokens,))
    #     # Should have an EOS token for every document.
    #     assert (memmap == tokenizer.eos_token_id).sum() == total_docs
    #     assert memmap[-1] == tokenizer.eos_token_id
    #     # Make sure all entries have been filled with actual token IDs.
    #     assert (memmap < tokenizer.vocab_size).all()
    #     log.info("All good!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    prepare_cli_environment()
    main()
