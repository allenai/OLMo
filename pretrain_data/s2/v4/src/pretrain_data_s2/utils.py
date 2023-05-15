"""'
how to run:

python process_text.py \
    src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc_clean/ \
    dst=... \
    cpu_count=1

"""

import datetime
import unicodedata
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from blingfire import text_to_words
from cached_path import cached_path

from .consts import GOOGLE_1T_CORPUS


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


def nfc_normalize(txt: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(txt, str):
        return unicodedata.normalize("NFC", txt)
    elif isinstance(txt, list):
        return [unicodedata.normalize("NFC", t) for t in txt]
    else:
        return txt


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
    title = str(row.get("title", "") or "")
    abstract = str(row.get("abstract", "") or "")
    paragraphs = row.get("filtered_paragraphs", []) or []  # pyright: ignore
    assert isinstance(paragraphs, list)
    # pyright: ignore
    return f"{title.strip()}\n{abstract.strip()}\n\n{' '.join(p.strip() for p in paragraphs)}"


def fix_missing_added(row: pd.Series) -> pd.Series:
    if pd.isna(row["added"]) or row["added"] == "":
        t = datetime.datetime.now(datetime.timezone.utc)
        row["added"] = t.isoformat(timespec="milliseconds") + "Z"
    return row


def fix_missing_created(row: pd.Series) -> pd.Series:
    if pd.isna(row["created"]) or row["created"] == "":
        year = 1 if pd.isna(row["year"]) else int(row["year"] or 1)
        row["created"] = f"{year:04d}-01-01T00:00:00.000Z"
    return row


def s2orc_merge_headers(all_paragraphs: List[dict]) -> List[str]:
    current_header = None
    new_paragraphs: List[str] = []
    for para in all_paragraphs:
        text = unicodedata.normalize("NFC", para["text"].strip())

        if para["type"] == "section_header":
            current_header = text

        elif para["type"] == "paragraph":
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
