from contextlib import ExitStack
from datetime import datetime
from functools import partial
import hashlib
from queue import Queue
import random
from threading import Thread
from time import sleep
from typing import List, Optional, Tuple
import springs as sp
from smashed.utils.io_utils import open_file_for_write
from cached_path import cached_path
import gzip
from multiprocessing import Manager, Pool, set_start_method
from tqdm import tqdm
import requests
import orjson as json
from uniseg.wordbreak import words as uniseg_get_words
import string


@sp.dataclass
class Config:
    src: str = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
    dst: str = 's3://ai2-llm/pretraining-data/sources/redpajama/'
    parallel: int = 1
    debug: bool = False
    raw: bool = False


def count_words(text: str) -> int:
    return sum(1 for word in uniseg_get_words(text) if not all(char in string.whitespace for char in word))


def download_single(paths: Tuple[str, str], pbar_queue: Optional[Queue] = None):
    src, dst = paths
    response = requests.get(src, stream=True)

    sleep(random.random())  # add a bit of delay between processes
    response = requests.get(src, stream=True)
    last_size = chunk_count = 0

    with open_file_for_write(dst, 'wb') as _g:
        g = gzip.open(_g, 'wb')

        for ln in response.iter_content(chunk_size=4096):
            g.write(ln)
            chunk_count += 1

            if pbar_queue and (chunk_count % 1000 == 0):
                current_size = g.tell()
                pbar_queue.put((0, 0, 0, current_size - last_size))
                last_size = current_size

    current_size = g.tell()
    pbar_queue.put((1, 0, 0, current_size - last_size)) if pbar_queue else None


def process_single(paths: Tuple[str, str], pbar_queue: Optional[Queue] = None):
    src, dst = paths
    response = requests.get(src, stream=True)
    last_size = 0

    sleep(random.random())  # add a bit of delay between processes

    added = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    created_guess = datetime(2023, 4, 17).strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    cnt_words, cnt_docs, cnt_chars = 0, 0, 0

    with open_file_for_write(dst, 'wb') as _g:
        g = gzip.open(_g, 'wb')

        for ln in response.iter_lines():
            row = json.loads(ln)
            text = row.pop('text')
            length = count_words(text)
            created = row.get('meta', {}).get('timestamp', created_guess)

            reshaped = {
                'id': hashlib.md5(row['meta']['url'].encode('utf-8')).hexdigest(),
                'text': text,
                'source': 'redpajama',
                'version': 'v0',
                'added': added,
                'created': created,
                'metadata': {**row.get('meta', {}), 'length': length},
            }
            json_reshaped = json.dumps(reshaped) + b'\n'
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
    with open(cached_path(cfg.src)) as f:
        all_src = [line.strip() for line in f]

    if cfg.raw:
        fn = download_single
        dst = cfg.dst.rstrip('/') + '/raw'
    else:
        fn = process_single
        dst = cfg.dst.rstrip('/') + '/v0'

    all_dst = [f'{dst}/{src.split("/")[-1]}.gzip' for src in all_src]

    if cfg.debug:
        with tqdm(total=len(all_src)) as pbar:
            for single_src, single_dst in zip(all_src, all_dst):
                fn((single_src, single_dst))
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

            for _ in pool.imap_unordered(
                partial(fn, pbar_queue=pbar_queue), tuple(zip(all_src, all_dst))
            ):
                ...

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()


if __name__ == '__main__':
    main()
