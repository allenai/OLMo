import gzip
import hashlib
import random
import string
from contextlib import ExitStack
from datetime import datetime
from functools import partial
from multiprocessing import Manager, Pool, set_start_method
from queue import Queue
from threading import Thread
from time import sleep
from typing import List, NamedTuple, Optional, Tuple

import orjson as json
import requests
import springs as sp
from cached_path import cached_path
from smashed.utils.io_utils import open_file_for_write, MultiPath, recursively_list_files
from tqdm import tqdm
from uniseg.wordbreak import words as uniseg_get_words


RP_RELEASE_TIME = "2023-04-17T11:00:00"
RP_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"



@sp.dataclass
class Config:
    src: str = "s3://ai2-llm/pretraining-data/sources/redpajama/raw/data/"
    dst: str = "s3://ai2-llm/pretraining-data/sources/redpajama/v0/"
    parallel: int = 1
    debug: bool = False
    raw: bool = False


def count_words(text: str) -> int:
    return sum(1 for word in uniseg_get_words(text) if not all(char in string.whitespace for char in word))


class Progress(NamedTuple):
    files: int
    docs: int
    words: int
    chars: int

    @classmethod
    def new(cls, f: int = 0, d: int = 0, w: int = 0, c: int = 0) -> "Progress":
        return cls(files=f, docs=d, words=w, chars=c)


def format_c4(row: dict):
    # c4 has the following keys
    # - text
    # - meta
    #   - timestamp
    #   - url
    #   - source
    #   - language

    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    row.pop('source')

    text = row.pop("text")
    length = count_words(text)
    created_string = row.get("meta", {}).get("timestamp", RP_RELEASE_TIME)
    created = datetime.strptime(created_string, RP_TIME_FORMAT).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama_c4",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def format_arxiv(row: dict):
    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    text = row.pop("text")
    length = count_words(text)
    created_string = row.get("meta", {}).get("timestamp", RP_RELEASE_TIME)
    created = datetime.strptime(created_string, RP_TIME_FORMAT).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    reshaped = {
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": "redpajama_arxiv",
        "version": "v0",
        "added": added,
        "created": created,
        "metadata": {**row.get("meta", {}), "length": length},
    }
    return reshaped


def process_single(paths: Tuple[str, str], pbar_queue: Optional[Queue] = None):
    src, dst = paths
    response = requests.get(src, stream=True)
    last_size = 0

    sleep(random.random())  # add a bit of delay between processes

    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    created_guess = datetime(2023, 4, 17).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    cnt_words, cnt_docs, cnt_chars = 0, 0, 0
    dst_cnt = 0

    with ExitStack() as input_stack, ExitStack() as output_stack:
        _input_stream = input_stack.enter_context(cached_path(src, "rb"))
        input_stream = input_stack.enter_context(gzip.open(_input_stream, "rt"))

        # _output_stream = output_stack.enter_context(open_file_for_write(dst, "wb"))
        # output_stream = output_stack.enter_context(gzip.open(_output_stream, "wt"))

        for ln in input_stream:
            data = json.loads(ln)
            import ipdb; ipdb.set_trace()

    with open_file_for_write(dst, "wb") as _g:
        g = gzip.open(_g, "wb")

        for ln in response.iter_lines():
            row = json.loads(ln)
            text = row.pop("text")
            length = count_words(text)
            created = row.get("meta", {}).get("timestamp", created_guess)

            reshaped = {
                "id": hashlib.md5(row["meta"]["url"].encode("utf-8")).hexdigest(),
                "text": text,
                "source": "redpajama",
                "version": "v0",
                "added": added,
                "created": created,
                "metadata": {**row.get("meta", {}), "length": length},
            }
            json_reshaped = json.dumps(reshaped) + b"\n"
            g.write(json_reshaped)

            if cnt_docs > 1000 and pbar_queue:
                current_size = g.tell()
                pbar_queue.put((0, cnt_docs, cnt_words, current_size - last_size))
                cnt_words, cnt_docs, cnt_chars = 0, 0, 0
                last_size = current_size

            cnt_words += length
            cnt_chars += len(json_reshaped)
            cnt_docs += 1

    if pbar_queue:
        current_size = g.tell()
        pbar_queue.put((1, cnt_docs, cnt_words, current_size - last_size))


def threaded_progressbar(q: Queue, timeout: float, total_files: Optional[int] = None):
    with ExitStack() as stack:
        files_pbar = stack.enter_context(tqdm(desc=" Files", unit="files", position=0, total=total_files))
        docs_pbar = stack.enter_context(tqdm(desc="  Docs", unit=" docs", position=1, unit_scale=True))
        tokens_pbar = stack.enter_context(tqdm(desc="Tokens", unit=" tokens", position=2, unit_scale=True))
        bytes_pbar = stack.enter_context(tqdm(desc="Size (bytes)", unit="B", position=3, unit_scale=True))
        while True:
            item = q.get()
            if item is None:
                break
            else:
                files, docs, tokens, chars = item
            files_pbar.update(files)
            docs_pbar.update(docs)
            tokens_pbar.update(tokens)
            bytes_pbar.update(chars)
            sleep(timeout)


@sp.cli(Config)
def main(cfg: Config):

    src, dst = MultiPath.parse(cfg.src), MultiPath.parse(cfg.dst)
    all_src = [MultiPath.parse(p) for p in recursively_list_files(src)]
    all_paths = [(str(src), str(dst / (p - src))) for p in all_src]

    if cfg.debug:
        with tqdm(total=len(all_src)) as pbar:
            for path in all_paths:
                process_single(path)
                pbar.update(1)
    else:
        set_start_method("spawn")

        with Pool(processes=cfg.parallel) as pool:
            pbar_queue: Queue = (manager := Manager()).Queue()
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
