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
import functools
import gzip
import itertools
import json
import logging
import multiprocessing as mp
import os
import random
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, List, Optional, Sequence, Tuple, TypeVar, Union

import click
import msgspec
import numpy as np
from cached_path import cached_path
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

T = TypeVar("T", bound=Sequence)


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


def tokenize_file(
    tokenizer: Tokenizer,
    path: str,
    safe_mode: bool = False,
    cache_dir: Optional[str] = None,
) -> Generator[List[int], None, None]:
    """Tokenize a file of documents using the provided tokenizer; file is expected to be a gzipped JSON lines
    file, each containing a field named `text`.
    """

    decoder = msgspec.json.Decoder(InputDocumentSpec)
    caching_path = path

    with ExitStack() as stack:
        if safe_mode:
            caching_path = cached_path(path, cache_dir=cache_dir)
            input_stream = stack.enter_context(gzip.open(caching_path, mode="rt"))
        else:
            input_file = stack.enter_context(stream_file_for_read(path, mode="rb"))
            input_stream = stack.enter_context(decompress_stream(input_file, mode="rt"))

        i = 1
        try:
            for line in input_stream:
                row = decoder.decode(line)
                if text := row.text.strip():
                    # skip empty docs
                    yield tokenizer.encode(text, add_special_tokens=True)
                i += 1
        except Exception as e:
            log.error(f"Error processing {path}:{i:,} -> {e}")
            pass

    if caching_path != path and os.path.exists(caching_path):
        os.remove(caching_path)


class MemmapFile:
    """Context manager responsible for writing, resizing, and closing / uploading a memmap file."""

    DEFAULT_MAX_TOKENS = 512 * 1024 * 1024  # 500M tokens / 1GB

    def __init__(
        self,
        path: str,
        dtype: np.dtype,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Create a new memmap file.

        Args:
            path (str): Location for the memmap file. If the path is not local, the memmap file will be
                written to a temporary file first and then uploaded to the destination.
            dtype (np.dtype): Data type for the memmap file; must be a valid numpy dtype.
            max_tokens (int, optional): Maximum number of tokens per file. Defaults to 500M tokens, which is 1GB.
        """
        self.path = MultiPath.parse(path)
        self.dtype = dtype
        self.max_tokens = max_tokens

        self._local_path: Optional[Path] = None
        self._written_tokens = 0
        self._memmap: Optional[np.memmap] = None

    def __len__(self) -> int:
        """Length of the memmap file in tokens that have been written."""
        return self._written_tokens

    def write(self, values: List[int], flush: bool = False) -> Optional[List[int]]:
        """Write a list of token IDs to the memmap file; if only a subset of the values can be written,
        return the rest.

        Args:
            values (List[int]): List of token IDs to write.
            flush (bool, optional): Whether to flush the memmap file after writing. Defaults to False.
        """

        if self._memmap is None:
            raise RuntimeError("MemmapFile is not open")

        if (len(values) + self._written_tokens) >= self.max_tokens:
            values = values[: self.max_tokens - self._written_tokens]
            rest = values[self.max_tokens - self._written_tokens :]
        else:
            rest = None

        self._memmap[self._written_tokens : self._written_tokens + len(values)] = values
        self._written_tokens += len(values)

        if flush:
            self._memmap.flush()

        return rest

    def __enter__(self) -> "MemmapFile":
        """Context manager entry point. Creates the memmap file and returns self."""

        assert self._memmap is None, "MemmapFile is already open"

        if self.path.is_local:
            self._local_path = self.path.as_path
            # make sure the directory exists
            self._local_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        else:
            with NamedTemporaryFile(delete=False, prefix="olmo_memmap") as f:
                # if the destination for the memmap is not local, we need to write to a temporary file first
                self._local_path = Path(f.name)

        self._memmap = np.memmap(mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self.max_tokens,))
        log.info(f"Created memmap file at {self._local_path} of size {self._memmap.nbytes:,} bytes")

        return self

    def __exit__(self, *_):
        """Context manager exit point. Closes the memmap file."""
        return self.close()

    def close(self):
        """Close the memmap file and optionally upload it to the destination (in the case of a remote path)."""
        assert self._local_path is not None, "MemmapFile is not open"
        assert self._memmap is not None, "MemmapFile is not open"

        try:
            # write the memmap to the destination
            self._memmap.flush()

            # we resize the memmap to the number of tokens actually written
            if self._written_tokens < self.max_tokens:
                del self._memmap
                os.rename(self._local_path, (temp_path := self._local_path.with_suffix(".tmp")))
                new_memmap = np.memmap(
                    mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self._written_tokens,)
                )
                old_memmap = np.memmap(mode="r", filename=temp_path, dtype=self.dtype, shape=(self.max_tokens,))
                new_memmap[:] = old_memmap[: self._written_tokens]
                new_memmap.flush()
                log.info(f"Resized memmap file from {old_memmap.nbytes:,} to {new_memmap.nbytes:,} bytes")
                os.remove(temp_path)

            if not self.path.is_local:
                with ExitStack() as stack:
                    f = stack.enter_context(stream_file_for_read(self._local_path, "rb"))
                    g = stack.enter_context(open_file_for_write(self.path, mode="wb"))
                    g.write(f.read())
                log.info(f"Written memmap file to {self.path.as_str}")
        finally:
            if not self.path.is_local:
                # delete the temporary file under any circumstances
                os.remove(self._local_path)

        self._local_path = self._memmap = None


def fill_memmap(
    tokenizer_id: str,
    path_or_paths: Union[str, List[str]],
    memmap_path: str,
    dtype: np.dtype,
    safe_mode: bool = False,
    max_tokens: int = 512 * 1024 * 1024,  # 512M tokens * 2 bytes per token (uint16) = 1GB
    sample_rate: float = 1.0,
    random_seed: int = 3920,
    repeat_sequence: int = 1,
    cache_dir: Optional[str] = None,
) -> int:
    """Write a memmap file from a file of documents."""

    # set the seed in case we need to sample
    np.random.seed(random_seed)

    # we need to make a new tokenizer here because it's not pickleable
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)

    # first memmap file will be created in the loop below
    memmap: Optional[MemmapFile] = None

    # we increment this every time we create a new memmap file
    file_index = 0

    # total number of tokens written
    total_tokens = 0

    # make sure path is a list
    path_or_paths = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths

    with ExitStack() as stack:
        it = itertools.chain.from_iterable(
            # repeat the sequence if necessary
            tokenize_file(tokenizer=tokenizer, path=path, safe_mode=safe_mode, cache_dir=cache_dir)
            for _ in range(repeat_sequence)
            for path in path_or_paths
        )
        for line_no, token_ids in enumerate(it, start=1):
            # perform sampling if necessary
            if sample_rate < 1.0 and np.random.rand() > sample_rate:
                continue

            # flush any 10k lines or so; improves stability
            flush = line_no % 10_000 == 0

            # increment the total number of tokens written
            total_tokens += len(token_ids)

            # if leftovers_to_write is not None it means that either memmap is None or it's full,
            # so we will need to create a new one later
            leftovers_to_write = memmap.write(token_ids, flush=flush) if memmap is not None else token_ids

            if leftovers_to_write is not None:
                # close the previous memmap (if one is open)
                stack.pop_all().close()

                # create a new memmap file; progressively name them with an index
                curr_memmap_path = f"{memmap_path}_{file_index:05d}.npy"
                memmap = stack.enter_context(MemmapFile(path=curr_memmap_path, dtype=dtype, max_tokens=max_tokens))

                # increment the file index and reset the tokens index
                file_index += 1

                # do the actual writing
                memmap.write(leftovers_to_write)

        # close the last memmap
        stack.pop_all().close()

    return total_tokens


def make_source_and_target(
    src: Tuple[str, ...],
    output: str,
    random_seed: int = 3920,
    paths_per_worker: int = 1,
) -> Tuple[Tuple[Union[str, List[str]], ...], Tuple[str, ...]]:
    """Recursively list all files in the source directories and create a corresponding list of destination."""

    np.random.seed(random_seed)
    random.seed(random_seed)

    exploded_src = list(set(path for prefix in src for path in recursively_list_files(prefix)))
    output_digits = np.ceil(np.log10(len(exploded_src) + 1)).astype(int)

    # shuffle the source paths
    random.shuffle(exploded_src)

    if paths_per_worker > 1:
        assert (
            len(exploded_src) >= paths_per_worker
        ), f"Number of paths ({len(exploded_src)}) must be >= paths_per_worker ({paths_per_worker})"

        # group the paths into chunks of paths_per_worker
        exploded_src = [
            sorted(exploded_src[i : i + paths_per_worker]) for i in range(0, len(exploded_src), paths_per_worker)
        ]

    # determine the destination paths
    exploded_dst = [f'{output.rstrip("/")}/{i:0{output_digits}d}' for i in range(len(exploded_src))]

    return tuple(exploded_src), tuple(exploded_dst)


@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=str,
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Specify the output path.",
    prompt="Output directory",
)
@click.option(
    "--tokenizer",
    "tokenizer_id",
    type=str,
    help="Name of path of a pretrained tokenizer",
    default="allenai/eleuther-ai-gpt-neox-20b-pii-special",
)
@click.option("--dtype", "dtype_str", default="uint16")
@click.option("--validate/--no-validate", default=False)
@click.option("--sample-rate", type=click.FloatRange(min=0.0, max=1.0), default=1.0)
@click.option("--random-seed", type=int, default=3920)
@click.option("--repeat-sequence", type=click.IntRange(min=1), default=1)
@click.option("--paths-per-worker", type=click.IntRange(min=1), default=1)
@click.option(
    "--cache-dir",
    type=str,
    default=None,
    help="Cache directory for the tokenizer; use system default if not specified",
)
@click.option(
    "--max-tokens",
    default=512 * 1024 * 1024,
    type=int,
    help="Maximum number of tokens to store in a single memmap file (default: 512M tokens or 1GB)",
)
@click.option("--debug/--no-debug", default=False, help="Enable debug (single process mode)")
@click.option(
    "--safe-mode/--fast-mode", default=False, help="Safe mode caches locally and decompresses using gzip.open"
)
@click.option("-j", "--workers", "max_workers", type=int, default=1, help="Defaults to number of CPUs")
def main(
    src: Tuple[str, ...],
    output: str,
    tokenizer_id: str = "EleutherAI/gpt-neox-20b",
    dtype_str: str = "uint16",
    validate: bool = False,
    max_tokens: int = 512 * 1024 * 1024,
    safe_mode: bool = False,
    debug: bool = False,
    sample_rate: float = 1.0,
    random_seed: int = 3920,
    repeat_sequence: int = 1,
    paths_per_worker: int = 1,
    max_workers: int = 1,
    cache_dir: Optional[str] = None,
):
    print("=== CONFIGURATION ===")
    print(f"src:              {src}")
    print(f"output:           {output}")
    print(f"tokenizer_id:     {tokenizer_id}")
    print(f"dtype_str:        {dtype_str}")
    print(f"validate:         {validate}")
    print(f"max_tokens:       {max_tokens}")
    print(f"safe_mode:        {safe_mode}")
    print(f"debug:            {debug}")
    print(f"sample_rate:      {sample_rate}")
    print(f"random_seed:      {random_seed}")
    print(f"repeat_sequence:  {repeat_sequence}")
    print(f"paths_per_worker: {paths_per_worker}")
    print(f"max_workers:      {max_workers}")
    print(f"cache_dir:        {cache_dir}")
    print("=====================")

    dtype = np.dtype(dtype_str)
    exploded_src, exploded_dst = make_source_and_target(
        src=src, output=output, random_seed=random_seed, paths_per_worker=paths_per_worker
    )

    # creating a partial here with all the arguments we need to pass to fill_memmap except for the paths
    # so that we don't make mistakes between debug and non-debug mode
    fill_memmap_fn = functools.partial(
        fill_memmap,
        tokenizer_id=tokenizer_id,
        dtype=dtype,
        max_tokens=max_tokens,
        safe_mode=safe_mode,
        sample_rate=sample_rate,
        random_seed=random_seed,
        repeat_sequence=repeat_sequence,
        cache_dir=cache_dir,
    )

    total_tokens_written = 0

    if debug:
        log.info("Running in debug mode. Only one process will be used.")
        for src_path, dst_path in zip(exploded_src, exploded_dst):
            total_tokens_written += fill_memmap_fn(path_or_paths=src_path, memmap_path=dst_path)
    else:
        # Now tokenizer all documents again and populate the memmap array. We do this in parallel.
        workers_cnt = min(max_workers or os.cpu_count() or 1, len(exploded_src))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_cnt) as executor:
            futures: List[Future[int]] = []
            for src_path, dst_path in zip(exploded_src, exploded_dst):
                future = executor.submit(fill_memmap_fn, path_or_paths=src_path, memmap_path=dst_path)
                futures.append(future)
            with get_progress() as progress:
                for future in progress.track(
                    concurrent.futures.as_completed(futures),
                    description="Filling memmap arrays...",
                    total=len(futures),
                ):
                    total_tokens_written += future.result()

    log.info(f"Done! File(s) written to {output}")
    log.info(f"Total tokens written: {total_tokens_written:,}")

    if validate:
        log.info("Validating...")
        tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)

        def encode_fn(row):
            return tokenizer.encode(json.loads(row)["text"], add_special_tokens=True)  # noqa

        total_tokens = total_docs = 0
        for input_path in (path for prefix in src for path in recursively_list_files(prefix)):
            with stream_file_for_read(input_path, mode="rb") as f, decompress_stream(f, mode="rt") as g:
                for row in g:
                    total_docs += 1
                    total_tokens += len(encode_fn(row))

        for output_path in recursively_list_files(output):
            memmap = np.memmap(output_path, mode="r", dtype=dtype)
            total_tokens -= len(memmap)
            total_docs -= (memmap == tokenizer.eos_token_id).sum()
            assert (memmap < tokenizer.vocab_size).all(), f"Invalid token ID in {output_path}"

        assert total_tokens == 0, f"Total tokens mismatch: {total_tokens} != 0"
        assert total_docs == 0, f"Total docs mismatch: {total_docs} != 0"

        log.info("All good!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    prepare_cli_environment()
    main()
