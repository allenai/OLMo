import gzip
import hashlib
import random
import re
import string
from contextlib import ExitStack
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Pool, set_start_method
from queue import Queue
from threading import Thread
from time import sleep
from typing import BinaryIO, List, NamedTuple, Optional, TextIO, Tuple, Union

import json
import zstandard
import requests
import springs as sp
from smashed.utils.io_utils import open_file_for_write, MultiPath, recursively_list_files, open_file_for_read
from tqdm import tqdm
from uniseg.wordbreak import words as uniseg_get_words


RP_RELEASE_TIME = "2023-04-17T11:00:00"
RP_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


@sp.dataclass
class Config:
    src: str = "s3://ai2-llm/pretraining-data/sources/redpajama/raw/data"
    dst: str = "s3://ai2-llm/pretraining-data/sources/redpajama/v0/"
    parallel: int = 1
    debug: bool = False
    raw: bool = False


def count_words(text: str) -> int:
    # length is calculated using a regex that splits on whitespace
    return re.sub(r"\s+", " ", text).count(' ')


class Progress(NamedTuple):
    files: int
    docs: int
    words: int
    chars: int

    @classmethod
    def new(cls, f: int = 0, d: int = 0, w: int = 0, c: int = 0) -> "Progress":
        return cls(files=f, docs=d, words=w, chars=c)


def format_c4(row: dict):

    text: str = row.pop("text")
    length = count_words(text)

    # c4 timestamp is '2019-12-31T23:59:59Z'; we strip the Z
    created_string = (row.get("meta", {}).get("timestamp", "") or RP_RELEASE_TIME).rstrip("Z")
    created = datetime.strptime(created_string, RP_TIME_FORMAT).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/c4",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def format_arxiv(row: dict):
    text: str = row.pop("text")
    length = count_words(text)

    # arxiv timestamp is '2019-12-31T23:59:59'
    created_string = row.get("meta", {}).get("timestamp", "") or RP_RELEASE_TIME
    created = datetime.strptime(created_string, RP_TIME_FORMAT).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/arxiv",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def format_github(row: dict):
    text: str = row.pop("text")
    length = count_words(text)

    # github has no date, so we use the release date
    created_string = RP_RELEASE_TIME
    created = datetime.strptime(created_string, RP_TIME_FORMAT).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/github",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


def format_wiki(row: dict):
    text: str = row.pop("text")
    length = count_words(text)

    # wiki only has date and it's YYYYMMDD, no '-'
    created_string = row.get("meta", {}).get("timestamp", "") or RP_RELEASE_TIME.split('T')[0].replace('-', '')
    created = datetime.strptime(created_string, '%Y%m%d').strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/wiki",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


def format_books(row: dict):
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    text: str = row.pop("text")
    length = count_words(text)

    # books only have year
    created_string = row.get("meta", {}).get("publication_date", "") or RP_RELEASE_TIME.split('-')[0]
    created = datetime.strptime(created_string, "%Y").strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/books",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


def format_stackexchange(row: dict):
    text: str = row.pop("text")
    length = count_words(text)

    # stack exchange only has date and it's YYYY-MM-DD
    created_string = row.get("meta", {}).get("timestamp", "") or RP_RELEASE_TIME.split('T')[0]
    created = datetime.strptime(created_string, '%Y-%m-%d').strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/stackexchange",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def format_commoncrawl(row: dict):
    text: str = row.pop("text")
    length = count_words(text)

    # common crawl is YYYY-WW (week of year)
    _, created_string, *_ = row['source'].split('/')
    created = datetime.strptime(f'{created_string}-0', '%Y-%W-%w').strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/commoncrawl",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


def process_single(paths: Tuple[str, str], pbar_queue: Optional['Queue[Union[None, Progress]]'] = None):
    src, dst = paths

    sleep(random.random())  # add a bit of delay between processes

    cnt_part = cnt_words = cnt_docs = 0

    dst_base, dst_fn = dst.rsplit('/', 1)
    dst_fn, dst_ext = dst_fn.split('.', 1)

    def _dst(p: int):
        return f'{dst_base}/{dst_fn}_{p:05d}.{dst_ext}'

    input_stream: TextIO
    output_stream: TextIO

    with ExitStack() as input_stack, ExitStack() as output_stack:
        _input_stream = input_stack.enter_context(open_file_for_read(src, "rb"))

        if src.endswith(".gz") or src.endswith(".gzip"):
            input_stream = input_stack.enter_context(gzip.open(_input_stream, "rt"))
        elif src.endswith(".zst"):
            input_stream = input_stack.enter_context(zstandard.open(_input_stream, "rt"))
        else:
            input_stream = input_stack.enter_context(open(_input_stream, "rt"))

        _output_stream = output_stack.enter_context(open_file_for_write(_dst(cnt_part), "wb"))
        output_stream = output_stack.enter_context(gzip.open(_output_stream, "wt"))     # pyright: ignore

        if 'common_crawl' in src:
            reshape_fn = format_commoncrawl
        elif 'arxiv' in src:
            reshape_fn = format_arxiv
        elif 'stackexchange' in src:
            reshape_fn = format_stackexchange
        elif 'books' in src:
            reshape_fn = format_books
        elif 'c4' in src:
            reshape_fn = format_c4
        elif 'wiki' in src:
            reshape_fn = format_wiki
        elif 'github' in src:
            reshape_fn = format_github
        else:
            raise ValueError(f'Unknown source: {src}')

        for i, ln in enumerate(input_stream):
            try:
                data = json.loads(ln)
            except Exception as e:
                raise ValueError(f"Error parsing {src}:{i}") from e

            reshaped = reshape_fn(data)
            output_stream.write(json.dumps(reshaped) + "\n")

            # split into 1GB files
            if output_stream.tell() > 1_000_000_000:
                output_stream.close()
                _output_stream.close()
                output_stack.pop_all().close()
                cnt_part += 1
                _output_stream = output_stack.enter_context(open_file_for_write(_dst(cnt_part), "wb"))
                output_stream = output_stack.enter_context(gzip.open(_output_stream, "wt"))     # pyright: ignore

            # update progress bar every 1000 docs
            if cnt_docs > 1000 and pbar_queue:
                pbar_queue.put(Progress.new(d=cnt_docs, w=cnt_words))
                cnt_words = cnt_docs = 0

            cnt_words += reshaped["metadata"]["length"]
            cnt_docs += 1

    if pbar_queue:
        pbar_queue.put(Progress.new(d=cnt_docs, w=cnt_words))


def threaded_progressbar(q: 'Queue[Union[None, Progress]]', timeout: float, total_files: Optional[int] = None):
    with ExitStack() as stack:
        files_pbar = stack.enter_context(tqdm(desc=" Files", unit="files", position=0, total=total_files))
        docs_pbar = stack.enter_context(tqdm(desc="  Docs", unit=" docs", position=1, unit_scale=True))
        tokens_pbar = stack.enter_context(tqdm(desc="Tokens", unit=" tokens", position=2, unit_scale=True))
        while True:
            item = q.get()
            if item is None:
                break

            files_pbar.update(item.files)
            docs_pbar.update(item.docs)
            tokens_pbar.update(item.words)

            sleep(timeout)


@sp.cli(Config)
def main(cfg: Config):

    src, dst = MultiPath.parse(cfg.src), MultiPath.parse(cfg.dst)
    files = recursively_list_files(src)

    files = [
        's3://ai2-llm/pretraining-data/sources/redpajama/raw/data/arxiv/arxiv_73241940-66c1-481c-b53a-f5e8b9afe9fa.jsonl.gz',
        # 's3://ai2-llm/pretraining-data/sources/redpajama/raw/data/book/book.jsonl.gz',
        # 's3://ai2-llm/pretraining-data/sources/redpajama/raw/data/stackexchange/stackexchange.jsonl.gz',
        # 's3://ai2-llm/pretraining-data/sources/redpajama/raw/data/wikipedia/wiki.jsonl.gz',
        's3://ai2-llm/pretraining-data/sources/redpajama/raw/data/c4/c4-train.00857-of-01024.jsonl.gz',
        's3://ai2-llm/pretraining-data/sources/redpajama/raw/data/github/filtered_216883d3a669406699428bc485a4c228.sampled.jsonl.gz',
        's3://ai2-llm/pretraining-data/sources/redpajama/raw/data/common_crawl/2023-06/en_middle_0104.json.gz.dedup.classifier.jsonl.zst',
    ]

    all_src = [MultiPath.parse(p) for p in files]
    all_paths = [(str(p), str(dst / (p - src))) for p in all_src]

    if cfg.debug:
        with tqdm(total=len(all_src)) as pbar:
            for path in all_paths:
                process_single(path)
                pbar.update(1)
    else:
        set_start_method("spawn")

        with Pool(processes=cfg.parallel) as pool:
            pbar_queue: 'Queue[Union[None, Progress]]' = (manager := Manager()).Queue()
            pbar_thread = Thread(
                target=threaded_progressbar,
                args=(pbar_queue, 0.1, len(all_src)),
                daemon=True,
            )
            pbar_thread.start()

            _process_single = partial(process_single, pbar_queue=pbar_queue)
            for _ in pool.imap_unordered(_process_single, all_paths):
                ...

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()


if __name__ == "__main__":
    main()
