"""'
how to run:

python process_text.py \
    src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc_clean/ \
    dst=... \
    cpu_count=1

"""

import datetime
import gc
import gzip
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
from typing import Any, Dict, List, Optional, Tuple, Union

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


DATA_COLUMNS = {"source", "id", "text", "added", "created", "metadata", "version"}


def row_to_metadata(row: pd.Series) -> Dict[str, Any]:
    return {col: row[col] for col in row.index if col not in DATA_COLUMNS}


def merge_text(row: pd.Series) -> str:
    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    paragraphs = row.get("filtered_paragraphs", []) or []
    return f"{title}\n\n{abstract}\n\n{' '.join(paragraphs)}"


def fix_missing_added(row: pd.Series) -> pd.Series:
    if pd.isna(row["added"]) or row["added"] == "":
        t = datetime.datetime.now(datetime.timezone.utc)
        row["added"] = t.isoformat(timespec="milliseconds") + "Z"
    return row


def fix_missing_created(row: pd.Series) -> pd.Series:
    if pd.isna(row["created"]) or row["created"] == "":
        year = int(row["year"] or 1)
        row["created"] = f"{year}-01-01T00:00:00.000Z"
    return row


def process_single(
    io_paths: Tuple[io_utils.MultiPath, io_utils.MultiPath],
    pbar_queue: Optional[Queue] = None,
    debug: bool = False,
):
    logger = sp.configure_logging(__name__, logging_level="WARNING", force_root_reattach=True)

    upp = UnigramPerplexityPredictor()
    src, dst = io_paths
    dst.path += ".gz"

    with io_utils.open_file_for_read(src, "rb", logger=logger) as f, NamedTemporaryFile("wb") as tmp:
        tmp.write(f.read())
        tmp.flush()
        df = pd.read_parquet(tmp.name)

    # for debugging purposes, only take first 1000 rows
    if debug:
        df = df.head(100)

    # filter all rows that don't have a "all_paragraphs" column
    df = df[df["all_paragraphs"].notna()]

    def merge_headers(all_paragraphs: List[dict]) -> List[str]:
        current_header = None
        new_paragraphs: List[str] = []
        for para in all_paragraphs:
            if para["type"] == "section_header":
                current_header = para["text"].strip()
            elif para["type"] == "paragraph":
                text = para["text"].strip()

                if current_header is not None:
                    text = f"\n{current_header}\n{text}"
                    current_header = None

                elif len(new_paragraphs) > 0:
                    # there is a previous paragraph, so we check to perform
                    # a few checks to make sure the current paragraph is
                    # not accidentally un-merged
                    pp = new_paragraphs[-1].rstrip()

                    if " " not in text:
                        # if the previous paragraph doesn't contain a space,
                        # then we merge the two paragraphs
                        text = new_paragraphs.pop(-1).rstrip() + " " + text
                    elif pp[-1] not in ".?!":
                        # if the previous paragraph doesn't end in a
                        # punctuation mark, then we merge the two paragraphs
                        text = new_paragraphs.pop(-1).rstrip() + " " + text
                    elif is_parenthetical_spanning_two_paragraphs(prev_para=pp, curr_para=text):
                        text = new_paragraphs.pop(-1).rstrip() + " " + text

                new_paragraphs.append(text)

        return new_paragraphs

    df["filtered_paragraphs"] = df["all_paragraphs"].apply(merge_headers)
    df.drop(columns=["all_paragraphs"], inplace=True)

    # assign version v0 and s2 as the source
    df["version"] = "v0"
    df["source"] = "s2"

    # fix missing added column
    df = df.apply(fix_missing_added, axis=1)

    # fix missing created column
    df = df.apply(fix_missing_created, axis=1)

    # spec requires id to be a string
    df["id"] = df["id"].astype(str)

    # create initial text by concatenating title and abstract and
    # all paragraphs
    df["text"] = df.apply(merge_text, axis=1)

    def get_language(filtered_paragraphs: List[str]) -> List[str]:
        langs: List[str] = []
        for para in filtered_paragraphs:
            try:
                text = para.strip()[:LANG_ID_CUT]
                langs.append(cld3.get_language(text).language)  # type: ignore
            except Exception:
                langs.append("unk")
        return langs

    # create new column that is the result of the function
    # cld3.get_language(text) applied to the text column
    df["language_paragraphs"] = df["filtered_paragraphs"].apply(get_language)

    # calculate the perplexity of each paragraph
    df["perplexity_paragraphs"] = df["filtered_paragraphs"].apply(lambda x: [upp.predict(para) for para in x])

    # zip the language, perplexity, and filtered paragraphs columns together
    df["paragraphs"] = df.apply(
        lambda x: list(
            dict(language=lang, perplexity=perp, text=para)
            for lang, perp, para in zip(
                x["language_paragraphs"], x["perplexity_paragraphs"], x["filtered_paragraphs"]
            )
        ),
        axis=1,
    )

    # get the number of tokens as a new column
    df["count"] = df["filtered_paragraphs"].apply(lambda x: sum(len(para.split()) for para in x))

    # get a frequency distribution of the tokens
    df["top_frequencies"] = df["filtered_paragraphs"].apply(
        lambda x: [
            {"token": k, "count": v}
            for k, v in
            # count using whitespace as a delimiter
            Counter(t for p in x for t in p.split()).most_common(COMMON_CUT)
        ]
    )

    # define a lambda function to cast to int or return -1
    # if the value is not a float or int
    df["year"] = df["year"].apply(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else -1)

    # drop after zipping
    df.drop(columns=["language_paragraphs", "perplexity_paragraphs", "filtered_paragraphs"], inplace=True)

    # put everything that is not part of the data spec in metadata
    df["metadata"] = df.apply(row_to_metadata, axis=1)
    cnt = int(sum(df["count"]))
    df = df.drop([c for c in df.columns if c not in DATA_COLUMNS], axis=1)

    with ExitStack() as stack:
        out_f = stack.enter_context(io_utils.open_file_for_write(dst, "wb"))
        out_stream = stack.enter_context(gzip.open(out_f, "wt"))
        for row in df.itertuples(index=False):
            content = json.dumps(row._asdict(), default=str).strip()
            out_stream.write(content + "\n")  # type: ignore

    if pbar_queue is not None:
        pbar_queue.put((1, int(len(df)), int(cnt)))


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
