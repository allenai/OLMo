"""'
how to run:

python process_text.py \
    src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc_clean/ \
    dst=... \
    cpu_count=1

"""

import gc
import os
from collections import Counter
from contextlib import ExitStack
from functools import partial
from multiprocessing import Manager, Pool, cpu_count, set_start_method
from queue import Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from time import sleep
from typing import List, Optional, Tuple, Union

import cld3
import numpy as np
import pandas as pd
import pyarrow as pa
import springs as sp
from blingfire import text_to_words
from cached_path import cached_path
from smashed.utils import io_utils
from tqdm import tqdm

LANG_ID_CUT = 2000
COMMON_CUT = 100
GOOGLE_1T_CORPUS = (
    "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv"
)


@sp.dataclass
class ProcessTextConfig:
    src: str = sp.field(default=sp.MISSING, help="Path to S3 prefix containing parqet files")
    dst: str = sp.field(default=sp.MISSING, help="Path to S3 prefix to write parqet files")
    debug: bool = sp.field(default=False, help="Debug mode")
    cpu_count: int = sp.field(default=cpu_count(), help="Number of processes to use")


class UnigramPerplexityPredictor:
    """Predicts the perplexity of a passage based on the unigram distribution
    probability of the words in a large corpus."""

    UNK = "<unk>"

    def __init__(self, word_counts_path: str = GOOGLE_1T_CORPUS):
        local_word_counts_path = cached_path(word_counts_path)
        with open(local_word_counts_path) as f:
            word_counts = {
                word: int(count) for word, count in (line.strip().split(",", 1) for line in f) if count.isnumeric()
            }

        word_total = sum(word_counts.values())
        word_total_log = np.log2(word_total)
        self.words_logp = {word: np.log2(count) - word_total_log for word, count in word_counts.items()}

        # <unk> token has fictional count of √vocab_size + 1
        self.words_logp[self.UNK] = np.log2(np.sqrt(len(self.words_logp)) + 1) - word_total_log

    def log_p(self, word: str) -> float:
        return self.words_logp.get(word.lower(), self.words_logp[self.UNK])

    def predict(self, text: Union[str, List[str]]) -> float:
        if isinstance(text, str):
            text = text_to_words(text).split()

        log_prob = sum(self.log_p(word) / len(text) for word in text)
        return log_prob


def is_parenthetical_spanning_two_paragraphs(
    prev_para: str, curr_para: str, open_sym: str = "(", clos_sym: str = ")"
) -> bool:
    """Checks if the previous paragraph ends with an open parenthesis and
    the current paragraph starts with a closing parenthesis. If so, then
    the two paragraphs are probably part of the same paragraph, so we
    return true."""

    if (open_prev := prev_para.rfind(open_sym)) < 0:
        # previous paragraph doesn't contain an open parenthesis
        return False

    if (clos_curr := curr_para.rfind(clos_sym)) < 0:
        # current paragraph doesn't contain a closing parenthesis
        return False

    if open_prev < prev_para.rfind(clos_sym):
        # previous paragraph contains a closing parenthesis after the last
        # open parenthesis, so the open parenthesis is not part of the
        # previous paragraph
        return False

    if (open_curr := curr_para.rfind(open_sym)) >= 0 and open_curr > clos_curr:
        # current paragraph contains an open parenthesis after the last
        # closing parenthesis, so the closing parenthesis is not part of
        # the current paragraph
        return False

    return True


def process_single(
    io_paths: Tuple[io_utils.MultiPath, io_utils.MultiPath],
    pbar_queue: Optional[Queue] = None,
    debug: bool = False,
):
    logger = sp.configure_logging(__name__, logging_level="WARNING", force_root_reattach=True)

    upp = UnigramPerplexityPredictor()
    src, dst = io_paths

    with io_utils.open_file_for_read(src, "rb", logger=logger) as f, NamedTemporaryFile("wb") as tmp:
        tmp.write(f.read())
        tmp.flush()
        df = pd.read_parquet(tmp.name)

    # for debugging purposes, only take first 1000 rows
    if debug:
        df = df.head(100)

    def get_language(text: str) -> str:
        try:
            return cld3.get_language(text.strip()).language  # type: ignore
        except Exception:
            return "unk"

    # create new column that is the result of the function
    # cld3.get_language(text) applied to the text column
    df["title_language"] = df["title"].apply(get_language)
    df["abstract_language"] = df["abstract"].apply(get_language)

    # calculate the perplexity of abstract
    df["title_perplexity"] = df["title"].apply(upp.predict)
    df["abstract_perplexity"] = df["abstract"].apply(upp.predict)

    # get the number of tokens as a new column
    df["title_count"] = df["title"].apply(lambda x: len(x.split()))
    df["abstract_count"] = df["abstract"].apply(lambda x: len(x.split()))

    # get the most common words in the title and abstract
    df["top_frequencies"] = (df["title"] + "\n\n" + df["abstract"]).apply(
        lambda x: [
            {"token": k, "count": v}
            for k, v in
            # count using whitespace as a delimiter
            Counter(t for t in x.split()).most_common(COMMON_CUT)
        ]
    )

    # define a lambda function to cast to int or return -1
    # if the value is not a float or int
    df["year"] = df["year"].apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else -1)

    schema = pa.schema(
        [
            ("id", pa.int32()),
            ("year", pa.int32()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("sha1", pa.string()),
            ("title_language", pa.string()),
            ("abstract_language", pa.string()),
            ("title_perplexity", pa.float64()),
            ("abstract_perplexity", pa.float64()),
            ("title_count", pa.int32()),
            ("abstract_count", pa.int32()),
            (
                "top_frequencies",
                pa.list_(
                    pa.struct(
                        [
                            ("token", pa.string()),
                            ("count", pa.int32()),
                        ]
                    )
                ),
            ),
        ]
    )

    # # write the dataframe to local parquet file
    with NamedTemporaryFile("wb", delete=False) as f:
        df.to_parquet((local_path := f.name), engine="pyarrow", schema=schema)

    # upload the parquet file to s3
    with io_utils.open_file_for_write(dst, "wb", logger=logger) as f, open(local_path, "rb") as tmp_read:
        f.write(tmp_read.read())

    # delete the local parquet file
    os.remove(local_path)

    if pbar_queue is not None:
        cnt = sum(df["title_count"] + df["abstract_count"])
        pbar_queue.put((1, int(len(df)), int(cnt)))

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

    src_paths = [io_utils.MultiPath.parse(p) for p in io_utils.recursively_list_files(src)]
    dst_paths = [dst / (diff) if len(diff := (single_src - src)) > 0 else dst for single_src in src_paths]

    if cfg.debug:
        with tqdm(total=len(src_paths)) as pbar:
            for single_src, single_dst in zip(src_paths, dst_paths):
                process_single((single_src, single_dst), debug=cfg.debug)
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
                partial(process_single, pbar_queue=pbar_queue, debug=cfg.debug), tuple(zip(src_paths, dst_paths))
            ):
                ...

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()


if __name__ == "__main__":
    main()
