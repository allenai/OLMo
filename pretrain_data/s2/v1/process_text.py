import gc
import json
import os
from collections import Counter
from contextlib import ExitStack
from functools import partial
from multiprocessing import Manager, Pool, cpu_count, set_start_method
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from time import sleep
from typing import Optional, Tuple

import cld3
import pandas as pd
import springs as sp
from smashed.utils import io_utils
from tqdm import tqdm

LANG_ID_CUT = 2000
COMMON_CUT = 100


@sp.dataclass
class ProcessTextConfig:
    src: str = sp.field(default=sp.MISSING, help="Path to S3 prefix containing parqet files")
    dst: str = sp.field(default=sp.MISSING, help="Path to S3 prefix to write parqet files")
    debug: bool = sp.field(default=False, help="Debug mode")
    cpu_count: int = sp.field(default=cpu_count(), help="Number of processes to use")


def process_single(io_paths: Tuple[io_utils.MultiPath, io_utils.MultiPath], pbar_queue: Optional[Queue] = None):
    logger = sp.configure_logging(__name__, logging_level="WARNING", force_root_reattach=True)

    src, dst = io_paths

    with io_utils.open_file_for_read(src, "rb", logger=logger) as f, NamedTemporaryFile("wb") as tmp:
        tmp.write(f.read())
        tmp.flush()
        df = pd.read_parquet(tmp.name)

    # df = pd.read_parquet(str(src))

    # # for debugging purposes, only take first 1000 rows
    # df = df.head(100)

    # strip leading and trailing whitespace
    df["text"] = df["text"].str.strip()

    def get_language(text: str) -> str:
        try:
            return cld3.get_language(text[:LANG_ID_CUT]).language  # type: ignore
        except Exception:
            return "unk"

    # create new column that is the result of the function
    # cld3.get_language(text) applied to the text column
    df["lang"] = df["text"].apply(get_language)

    # whitespace tokenize the text column
    df["tokens"] = df["text"].str.split()

    # get the number of tokens as a new column
    df["cnt"] = df["tokens"].apply(len)

    def get_freqs_as_json(tokens: list) -> str:
        # gotta store as a json string because parquet doesn't support
        # dicts as is.
        return json.dumps(Counter(tokens).most_common(COMMON_CUT))

    # get a frequency distribution of the tokens
    df["freq"] = df["tokens"].apply(get_freqs_as_json)

    # drop the tokens column
    df = df.drop(columns=["tokens"])

    # # write the dataframe to the destination
    # df.to_parquet(str(dst))

    with NamedTemporaryFile("wb", delete=False) as tmp_write:
        df.to_parquet((tmp_name := tmp_write.name))

    # dst = str(dst).replace('=', '_')
    with io_utils.open_file_for_write(dst, "wb", logger=logger) as f, open(tmp_name, "rb") as tmp_read:
        f.write(tmp_read.read())

    os.remove(tmp_name)

    if pbar_queue is not None:
        pbar_queue.put((1, int(len(df)), int(sum(df["cnt"]))))

    del df
    gc.collect()


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


@sp.cli(ProcessTextConfig)
def main(cfg: ProcessTextConfig):
    src = io_utils.MultiPath.parse(cfg.src)
    dst = io_utils.MultiPath.parse(cfg.dst)

    src_paths = [p for p in io_utils.recursively_list_files(src)]
    dst_paths = [dst / (single_src - src) for single_src in src_paths]

    if cfg.debug:
        with tqdm(total=len(src_paths)) as pbar:
            for single_src, single_dst in zip(src_paths, dst_paths):
                process_single((single_src, single_dst))
                pbar.update(1)

    else:
        set_start_method("spawn")

        with Pool(processes=cfg.cpu_count) as pool:
            pbar_queue: Queue = (manager := Manager()).Queue()
            pbar_thread = Thread(
                target=threaded_progressbar,
                args=(pbar_queue, 0.1, len(src_paths)),
                daemon=True,
            )
            pbar_thread.start()

            for _ in pool.imap_unordered(
                partial(process_single, pbar_queue=pbar_queue), tuple(zip(src_paths, dst_paths))
            ):
                ...

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()


if __name__ == "__main__":
    main()
