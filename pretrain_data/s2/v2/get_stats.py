"""'
how to run:

python process_text.py \
    src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc_clean/ \
    dst=... \
    cpu_count=1

"""

import gzip
import string
from contextlib import ExitStack
from functools import partial
from multiprocessing import Manager, Pool, cpu_count, set_start_method
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Optional

import orjson as json

# import cld3
# import numpy as np
# import pandas as pd
# import pyarrow as pa
import springs as sp

# from blingfire import text_to_words
# from cached_path import cached_path
from smashed.utils import io_utils
from tqdm import tqdm
from uniseg.wordbreak import words as uniseg_get_words


@sp.dataclass
class ProcessTextConfig:
    src: str = sp.field(default=sp.MISSING, help="Path to S3 prefix containing parqet files")
    debug: int = sp.field(default=0, help="Debug mode. Set to >0 to enable")
    parallel: int = sp.field(default=cpu_count(), help="Number of processes to use")


def log(files: int = 0, docs: int = 0, tokens: int = 0, queue: Optional[Queue] = None):
    if queue is not None:
        queue.put((files, docs, tokens))
    else:
        print(f"Files: {files:,}, Docs: {docs:,}, Tokens: {tokens:,}")


def process_single(
    src: io_utils.MultiPath,
    pbar_queue: Optional[Queue] = None,
):
    logger = sp.configure_logging(__name__, logging_level="WARNING", force_root_reattach=True)

    log_fn = partial(log, queue=pbar_queue)

    total_docs_cnt = total_tokens_cnt = 0
    docs_cnt = tokens_cnt = 0

    with ExitStack() as stack:
        f = stack.enter_context(io_utils.open_file_for_read(src, "rb", logger=logger))
        stream = stack.enter_context(gzip.open(f, "rt"))

        for line in stream:
            data = json.loads(line)
            docs_cnt += 1
            tokens_cnt += sum(
                1 for word in uniseg_get_words(data["text"]) if not all(char in string.whitespace for char in word)
            )

            if docs_cnt > 1_000:
                log_fn(docs=docs_cnt, tokens=tokens_cnt)
                total_docs_cnt += docs_cnt
                total_tokens_cnt += tokens_cnt
                docs_cnt = tokens_cnt = 0

    if docs_cnt > 0 and pbar_queue is not None:
        log_fn(files=1, docs=docs_cnt, tokens=tokens_cnt)
        total_docs_cnt += docs_cnt
        total_tokens_cnt += tokens_cnt

    return total_docs_cnt, total_tokens_cnt


def threaded_progressbar(
    q: Queue,
    total_files: Optional[int] = None,
    timeout: float = 0.01,
):
    with ExitStack() as stack:
        files_pbar = stack.enter_context(tqdm(desc=" Files", unit="files", position=0, total=total_files))
        docs_pbar = stack.enter_context(tqdm(desc="  Docs", unit=" docs", position=1, unit_scale=True))
        tokens_pbar = stack.enter_context(tqdm(desc="Tokens", unit=" tokens", position=2, unit_scale=True))
        while True:
            try:
                item = q.get_nowait()
            except Empty:
                sleep(timeout)
                continue
            if item is None:
                break
            else:
                files, docs, tokens = item
            files_pbar.update(files)
            docs_pbar.update(docs)
            tokens_pbar.update(tokens)
            sleep(timeout)


@sp.cli(ProcessTextConfig)
def main(cfg: ProcessTextConfig):
    src = io_utils.MultiPath.parse(cfg.src)

    docs_cnt = tokens_cnt = 0

    src_paths = [io_utils.MultiPath.parse(p) for p in io_utils.recursively_list_files(src)]

    if cfg.debug > 0:
        src_paths = src_paths[: cfg.debug]
        with tqdm(total=len(src_paths)) as pbar:
            for single_src in src_paths:
                single_docs_cnt, single_tokens_cnt = process_single(single_src)
                docs_cnt += single_docs_cnt
                tokens_cnt += single_tokens_cnt
                pbar.update(1)

    else:
        set_start_method("spawn")

        with Pool(processes=cfg.parallel) as pool:
            pbar_queue: Queue = (manager := Manager()).Queue()
            pbar_thread = Thread(
                target=threaded_progressbar,
                args=(pbar_queue, len(src_paths)),
                daemon=True,
            )
            pbar_thread.start()

            for single_docs_cnt, single_tokens_cnt in pool.imap_unordered(
                partial(process_single, pbar_queue=pbar_queue), src_paths
            ):
                docs_cnt += single_docs_cnt
                tokens_cnt += single_tokens_cnt

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()

    print(f"Total files:  {len(src_paths):,}")
    print(f"Total docs:   {docs_cnt:,}")
    print(f"Total tokens: {tokens_cnt:,}")


if __name__ == "__main__":
    main()
