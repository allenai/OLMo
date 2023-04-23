import gzip
import hashlib
import json
import random
import re
from contextlib import ExitStack
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Pool, set_start_method
from queue import Queue
from threading import Thread
from time import sleep
from typing import NamedTuple, Optional, TextIO, Union

import springs as sp
import zstandard
from smashed.utils.io_utils import (
    MultiPath,
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)
from tqdm import tqdm


RP_RELEASE_TIME = "2023-04-17T11:00:00"
RP_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


@sp.dataclass
class Config:
    src: str = "s3://ai2-llm/pretraining-data/sources/redpajama/raw/data"
    dst: str = "s3://ai2-llm/pretraining-data/sources/redpajama/v0/documents"
    single: Optional[str] = sp.field(default=None, help="Only process this file")
    parallel: int = 1
    debug: bool = False
    raw: bool = False
    dryrun: bool = False
    version: str = 'v0'


def count_words(text: str) -> int:
    # length is calculated using a regex that splits on whitespace
    return re.sub(r"\s+", " ", text).count(" ")


class Progress(NamedTuple):
    files: int
    docs: int
    words: int
    chars: int

    @classmethod
    def new(cls, f: int = 0, d: int = 0, w: int = 0, c: int = 0) -> "Progress":
        return cls(files=f, docs=d, words=w, chars=c)


def format_c4(row: dict, version: str):
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
        "version": version,
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def format_arxiv(row: dict, version: str):
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
        "version": version,
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def format_github(row: dict, version: str):
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
        "version": version,
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


def format_wiki(row: dict, version: str):
    text: str = row.pop("text")
    length = count_words(text)

    # wiki only has date and it's YYYYMMDD, no '-'
    created_string = row.get("meta", {}).get("timestamp", "") or RP_RELEASE_TIME.split("T")[0].replace("-", "")
    created = datetime.strptime(created_string, "%Y%m%d").strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/wikipedia",
        "version": version,
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


def format_books(row: dict, version: str):
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    text: str = row.pop("text")
    length = count_words(text)

    # books only have year
    created_string = str(row.get("meta", {}).get("publication_date", "")) or RP_RELEASE_TIME.split("-")[0]
    created = datetime.strptime(created_string, "%Y").strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/books",
        "version": version,
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


def format_stackexchange(row: dict, version: str):
    text: str = row.pop("text")
    length = count_words(text)

    # stack exchange only has date and it's YYYY-MM-DD
    created_string = row.get("meta", {}).get("timestamp", "") or RP_RELEASE_TIME.split("T")[0]
    created = datetime.strptime(created_string, "%Y-%m-%d").strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/stackexchange",
        "version": version,
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def format_commoncrawl(row: dict, version: str):
    text: str = row.pop("text")
    length = count_words(text)

    # common crawl is YYYY-WW (week of year)
    _, created_string, *_ = row["source"].split("/")
    created = datetime.strptime(f"{created_string}-0", "%Y-%W-%w").strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama/commoncrawl",
        "version": version,
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }

    return reshaped


class ProcessPath(NamedTuple):
    src_prefix: str
    dst_prefix: str
    src_filename: str
    dst_filename: str

    @property
    def src(self):
        return f'{self.src_prefix.rstrip("/")}/{self.src_filename.lstrip("/")}'

    @property
    def fn_clean(self):
        fn = self.dst_filename.lstrip('/')

        if fn.startswith('common_crawl/'):
            # format: common_crawl/2019-30/en_head_0000.json.gz.dedup.classifier.jsonl.zst
            dataset, date, fn = fn.split('/', 2)
            fn = f'dataset={dataset}/{date}_{fn}'
        else:
            dataset, fn = fn.split('/', 1)
            fn = f'dataset={dataset}/{fn}'

        if fn.endswith('.zst'):
            fn = f'{fn[:-4]}.gz'

        return fn

    @property
    def pfx_clean(self):
        return self.dst_prefix.rstrip("/")

    @property
    def train(self):
        return f'{self.pfx_clean}/split=train/{self.fn_clean}'

    @property
    def test(self):
        return f'{self.pfx_clean}/split=test/{self.fn_clean}'

    @property
    def valid(self):
        return f'{self.pfx_clean}/split=valid/{self.fn_clean}'

    @classmethod
    def parse(cls, src: str, src_prefix: str, dst_prefix: str):
        dst_fn = src_fn = (MultiPath.parse(src) - MultiPath.parse(src_prefix)).as_str
        return cls(src_prefix, dst_prefix, src_fn, dst_fn)


def process_single(
    path: ProcessPath,
    version: str,
    pbar_queue: Optional["Queue[Union[None, Progress]]"] = None,
    dryrun: bool = False
):
    cnt_part = cnt_words = cnt_docs = 0

    def _dst(p: int, d: str):
        dst_base, dst_fn = d.rsplit("/", 1)
        dst_fn, dst_ext = dst_fn.split(".jsonl", 1)
        return f"{dst_base}/{dst_fn}_{p:05X}.jsonl{dst_ext}"

    if "common_crawl/" in path.src:
        reshape_fn = format_commoncrawl
    elif "arxiv/" in path.src:
        reshape_fn = format_arxiv
    elif "stackexchange/" in path.src:
        reshape_fn = format_stackexchange
    elif "book/" in path.src:
        reshape_fn = format_books
    elif "c4/" in path.src:
        reshape_fn = format_c4
    elif "wikipedia/" in path.src:
        reshape_fn = format_wiki
    elif "github/" in path.src:
        reshape_fn = format_github
    else:
        raise ValueError(f"Unknown source: {path.src}")

    input_stream: TextIO
    train_stream: TextIO

    if dryrun:
        pbar_queue.put(Progress.new(f=1)) if pbar_queue else None
        print(f'Processing "{path.src}" to "{path.train}" (and valid/test)')
        return

    # add a bit of delay between processes
    sleep(random.random() * 5.0)

    with ExitStack() as single_streams, ExitStack() as part_streams:
        _input_stream = single_streams.enter_context(open_file_for_read(path.src, "rb"))

        if path.src.endswith(".gz") or path.src.endswith(".gzip"):
            input_stream = single_streams.enter_context(gzip.open(_input_stream, "rt"))  # pyright: ignore
        elif path.src.endswith(".zst"):
            input_stream = single_streams.enter_context(zstandard.open(_input_stream, "rt"))  # pyright: ignore
        else:
            input_stream = single_streams.enter_context(open(_input_stream, "rt"))  # pyright: ignore

        _train_stream = part_streams.enter_context(open_file_for_write(_dst(p=cnt_part, d=path.train), "wb"))
        train_stream = part_streams.enter_context(gzip.open(_train_stream, "wt"))  # pyright: ignore

        _valid_stream = single_streams.enter_context(open_file_for_write(path.valid, "wb"))
        valid_stream = single_streams.enter_context(gzip.open(_valid_stream, "wt"))  # pyright: ignore

        _test_stream = single_streams.enter_context(open_file_for_write(path.test, "wb"))
        test_stream = single_streams.enter_context(gzip.open(_test_stream, "wt"))  # pyright: ignore

        i = 0
        while True:
            try:
                ln = next(input_stream)
            except StopIteration:
                break
            except Exception as e:
                print(f"\n\nError parsing {path.src}:{i} ({e})\n\n")
                continue

            i += 1

            try:
                data = json.loads(ln.encode("utf-8", "ignore").decode("utf-8"))
            except Exception as e:
                print(f"\n\nJSON Error parsing {path.src}:{i} ({e})\n\n")
                continue

            reshaped = reshape_fn(row=data, version=version)

            if reshaped["id"][:3] in {"fff", "ffe"}:
                test_stream.write(json.dumps(reshaped) + "\n")      # pyright: ignore
            elif reshaped["id"][:3] in {"ffd", "ffc"}:
                valid_stream.write(json.dumps(reshaped) + "\n")     # pyright: ignore
            else:
                train_stream.write(json.dumps(reshaped) + "\n")

            # split into 1GB files
            if _train_stream.tell() > 1_000_000_000:
                train_stream.close()
                _train_stream.close()
                part_streams.pop_all().close()
                cnt_part += 1
                _train_stream = part_streams.enter_context(
                    open_file_for_write(_dst(p=cnt_part, d=path.train), "wb")
                )
                train_stream = part_streams.enter_context(gzip.open(_train_stream, "wt"))  # pyright: ignore

            # update progress bar every 1000 docs
            if cnt_docs > 1000 and pbar_queue:
                pbar_queue.put(Progress.new(d=cnt_docs, w=cnt_words))
                cnt_words = cnt_docs = 0

            cnt_words += reshaped["metadata"]["length"]
            cnt_docs += 1

    if pbar_queue:
        pbar_queue.put(Progress.new(f=1, d=cnt_docs, w=cnt_words))


def threaded_progressbar(q: "Queue[Union[None, Progress]]", timeout: float, total_files: Optional[int] = None):
    with ExitStack() as stack:
        files_pbar = stack.enter_context(tqdm(desc=" Files", unit="f", position=0, total=total_files))
        docs_pbar = stack.enter_context(tqdm(desc="  Docs", unit="d", position=1, unit_scale=True))
        tokens_pbar = stack.enter_context(tqdm(desc="Tokens", unit="t", position=2, unit_scale=True))
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
    random.seed(0)

    if cfg.single:
        all_paths = [ProcessPath.parse(src=cfg.single, src_prefix=cfg.src, dst_prefix=cfg.dst)]
    else:
        all_paths = [
            ProcessPath.parse(src=p, src_prefix=cfg.src, dst_prefix=cfg.dst)
            for p in recursively_list_files(cfg.src)
        ]
        random.shuffle(all_paths)

    if cfg.debug:
        with tqdm(total=len(all_paths)) as pbar:
            for path in all_paths:
                process_single(path=path, version=cfg.version, dryrun=cfg.dryrun)
                pbar.update(1)
    else:
        set_start_method("spawn")

        with Pool(processes=cfg.parallel) as pool:
            pbar_queue: "Queue[Union[None, Progress]]" = (manager := Manager()).Queue()
            pbar_thread = Thread(
                target=threaded_progressbar,
                args=(pbar_queue, 0.01, len(all_paths)),
                daemon=True,
            )
            pbar_thread.start()

            _process_single = partial(process_single, pbar_queue=pbar_queue, version=cfg.version, dryrun=cfg.dryrun)
            for _ in pool.imap_unordered(_process_single, all_paths):
                ...

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()


if __name__ == "__main__":
    main()
