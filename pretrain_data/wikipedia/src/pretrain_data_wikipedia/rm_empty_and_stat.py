import gzip
import json
import string
from contextlib import ExitStack
from functools import partial, reduce
from multiprocessing import Manager, Pool, current_process, set_start_method
from queue import Queue
from threading import Thread
from time import sleep
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import springs as sp
from cached_path import cached_path
from smashed.utils.io_utils import (
    MultiPath,
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)
from tqdm import tqdm
from uniseg.wordbreak import words as uniseg_get_words

GOOGLE_1T_CORPUS = (
    "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/"
    "lucas/google-1T-unigram/unigram_freq.csv"
)


class PbarMsg(NamedTuple):
    started_files: int
    done_files: int
    done_docs: int
    done_tokens: int

    @classmethod
    def start(cls):
        return cls(1, 0, 0, 0)

    @classmethod
    def complete(cls):
        return cls(0, 1, 0, 0)

    @classmethod
    def prog(cls, docs: int, tokens: int):
        return cls(0, 0, docs, tokens)


class UnigramPerplexityPredictor:
    """Predicts the perplexity of a passage based on the unigram distribution
    probability of the words in a large corpus."""

    UNK = "<unk>"

    def __init__(self, word_counts_path: str = GOOGLE_1T_CORPUS):
        local_word_counts_path = cached_path(word_counts_path)
        with open(local_word_counts_path) as f:
            word_counts = {
                word: int(count)
                for word, count in (line.strip().split(",", 1) for line in f)
                if count.isnumeric()
            }

        word_total = sum(word_counts.values())
        word_total_log = np.log2(word_total)
        self.words_logp = {
            word: np.log2(count) - word_total_log
            for word, count in word_counts.items()
        }

        # <unk> token has fictional count of √vocab_size + 1
        self.words_logp[self.UNK] = (
            np.log2(np.sqrt(len(self.words_logp)) + 1) - word_total_log
        )

    def log_p(self, word: str) -> float:
        return self.words_logp.get(word.lower(), self.words_logp[self.UNK])

    def predict(self, text: List[str]) -> float:
        log_prob = sum(self.log_p(word) / len(text) for word in text)
        return log_prob


def get_current_process_number() -> int:
    if not (pid := current_process()._identity):
        return 0
    return reduce(lambda x, y: x * y, pid, 1)


def get_words(text: str) -> List[str]:
    return [
        word
        for word in uniseg_get_words(text)
        if not all(char in string.whitespace for char in word)
    ]


def process_single(
    paths: Tuple[MultiPath, MultiPath],
    pbar_queue: Optional[Queue] = None,
    tmp_dir: Optional[str] = None,
):
    src, dst = map(str, paths)

    ug = UnigramPerplexityPredictor()

    # count file as started
    pbar_queue.put(PbarMsg.start()) if pbar_queue else None
    docs_cnt = tokens_cnt = 0

    with ExitStack() as stack:
        in_f = stack.enter_context(
            open_file_for_read(src, "rb", temp_dir=tmp_dir)
        )
        in_stream = stack.enter_context(gzip.open(in_f, "rt"))
        out_f = stack.enter_context(
            open_file_for_write(dst, "wb", temp_dir=tmp_dir)
        )
        out_stream = stack.enter_context(gzip.open(out_f, "wt"))

        for raw in in_stream:
            if not (raw := raw.strip()):
                continue

            doc = json.loads(raw)

            if not (text := doc.get("text", "")):
                continue

            doc["words"] = get_words(text)
            doc["perplexity"] = ug.predict(doc["words"])

            out = json.dumps(doc) + "\n"
            out_stream.write(out)  # type: ignore

            docs_cnt += 1
            tokens_cnt += len(doc["words"])

            # update the progress bar every 100 docs
            if pbar_queue and docs_cnt >= 1000:
                pbar_queue.put(PbarMsg.prog(docs_cnt, tokens_cnt))
                docs_cnt = tokens_cnt = 0

    # one final update for the progress bar
    pbar_queue.put(PbarMsg.prog(docs_cnt, tokens_cnt)) if pbar_queue else None

    # count file as done
    pbar_queue.put(PbarMsg.complete()) if pbar_queue else None


@sp.dataclass
class Config:
    src: str = sp.MISSING
    dst: str = sp.MISSING
    parallel: int = 1
    debug: bool = False
    tmp_dir: Optional[str] = None


def threaded_progressbar(
    q: Queue, timeout: float, total_files: Optional[int] = None
):
    with ExitStack() as stack:
        files_started_pbar = stack.enter_context(
            tqdm(desc="Started", unit=" files", position=0, total=total_files)
        )
        files_completed_pbar = stack.enter_context(
            tqdm(desc="Done", unit=" files", position=1, total=total_files)
        )
        docs_pbar = stack.enter_context(
            tqdm(desc="  Docs", unit=" docs", position=2, unit_scale=True)
        )
        tokens_pbar = stack.enter_context(
            tqdm(desc="Tokens", unit=" toks", position=3, unit_scale=True)
        )
        while True:
            item = q.get()
            if item is None:
                break
            else:
                item = PbarMsg(*item)
            files_started_pbar.update(item.started_files)
            docs_pbar.update(item.done_docs)
            tokens_pbar.update(item.done_tokens)
            files_completed_pbar.update(item.done_files)
            sleep(timeout)


@sp.cli(Config)
def main(cfg: Config):
    src = MultiPath.parse(cfg.src)
    dst = MultiPath.parse(cfg.dst)
    src_paths = [MultiPath.parse(p) for p in recursively_list_files(src)]
    dst_paths = [dst / (p - src) for p in src_paths]

    if cfg.debug:
        with tqdm(total=len(src_paths)) as pbar:
            for single_src, single_dst in zip(src_paths, dst_paths):
                process_single((single_src, single_dst), tmp_dir=cfg.tmp_dir)
                pbar.update(1)

    else:
        set_start_method("spawn")

        with Pool(processes=cfg.parallel) as pool:
            pbar_queue: Queue = (manager := Manager()).Queue()
            pbar_thread = Thread(
                target=threaded_progressbar,
                args=(pbar_queue, 1 / 60, len(src_paths)),
                daemon=True,
            )
            pbar_thread.start()

            for _ in pool.imap_unordered(
                partial(
                    process_single, pbar_queue=pbar_queue, tmp_dir=cfg.tmp_dir
                ),
                tuple(zip(src_paths, dst_paths)),
            ):
                pass

            pool.close()
            pool.join()

            pbar_queue.put(None)
            pbar_thread.join()
            manager.shutdown()


if __name__ == "__main__":
    main()
