from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from multiprocessing import Manager
from threading import Thread
from cached_path import cached_path
from smashed.utils.io_utils import open_file_for_write, open_file_for_read
import gzip
from datetime import datetime
import json
from hashlib import md5
import os
from tqdm import tqdm
from queue import Queue
from typing import Optional, List
from time import sleep
from uniseg.wordbreak import words as uniseg_get_words
import string


S3_DST = 's3://ai2-llm/pretraining-data/sources/c4/v0/documents'
TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
DATA_URL = (
    "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/"
    "{name}/c4-{split}.{index:05d}-of-{n_shards:05d}.json.gz"
)
NAME = 'en'
SPLIT = 'train'
N_SHARDS = 1024


def get_words(text: str) -> List[str]:
    return [word for word in uniseg_get_words(text) if not all(char in string.whitespace for char in word)]


def threaded_progressbar(q: Queue, timeout: float, total_files: Optional[int] = None):
    with ExitStack() as stack:
        files_pbar = stack.enter_context(tqdm(desc=" Files", unit="files", position=0, total=total_files))
        docs_pbar = stack.enter_context(tqdm(desc="  Docs", unit=" docs", position=1, unit_scale=True))
        tokens_pbar = stack.enter_context(tqdm(desc="Tokens", unit=" tokens", position=2, unit_scale=True))
        while True:
            item = q.get()
            if item is None:
                break
            else:
                files, docs, tokens = item
            files_pbar.update(files)
            docs_pbar.update(docs)
            tokens_pbar.update(tokens)
            sleep(timeout)


def process_single(src_path, dst_path, queue: Optional[Queue] = None):
    with ExitStack() as stack:
        src_path = cached_path(src_path, quiet=True)
        input_file = stack.enter_context(open_file_for_read(src_path, 'rb'))
        input_stream = stack.enter_context(gzip.open(input_file, 'rt'))

        output_file = stack.enter_context(open_file_for_write(dst_path, 'wb'))
        output_stream = stack.enter_context(gzip.open(output_file, 'wt'))

        docs_cnt, tokens_cnt = 0, 0

        for row in input_stream:
            doc = json.loads(row)
            data = {
                'text': doc['text'],
                'id': md5(doc['url'].encode('utf-8')).hexdigest(),
                "version" : "v0",
                "source" : "c4",
                'added': TIMESTAMP,
                'created': doc['timestamp'],
                'metadata': {
                    'url': doc['url'],
                    'lang': NAME,
                    'length': len(get_words(doc['text'])),
                    'split': SPLIT,
                }
            }
            output_stream.write(json.dumps(data) + '\n')    # type: ignore
            docs_cnt += 1
            tokens_cnt += data['metadata']['length']

            if docs_cnt > 1000:
                queue.put((0, docs_cnt, tokens_cnt)) if queue else None
                docs_cnt, tokens_cnt = 0, 0

        os.remove(src_path)
        queue.put((1, docs_cnt, tokens_cnt)) if queue else None


def main():
    paths = [
        (
            DATA_URL.format(name=NAME, split=SPLIT, index=i, n_shards=N_SHARDS),
            f'{S3_DST}/{SPLIT}/part_{i:04d}.jsonl.gz'
        ) for i in range(N_SHARDS)
    ]

    with ProcessPoolExecutor(max_workers=60) as pool:
        pbar_queue: Queue = (manager := Manager()).Queue()
        pbar_thread = Thread(
            target=threaded_progressbar,
            args=(pbar_queue, 1 / 60, len(paths)),
            daemon=True,
        )
        pbar_thread.start()
        futures = []
        for src_path, dst_path in paths:
            futures.append(pool.submit(process_single, src_path, dst_path, queue=pbar_queue))

        while any(not future.done() for future in futures):
            sleep(1)

        pbar_queue.put(None)
        pbar_thread.join()
        manager.shutdown()

    # for src_path, dst_path in paths:
    #     process_single(src_path, dst_path)


if __name__ == '__main__':
    main()
