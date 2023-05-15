"""'
how to run:

python process_text.py \
    src=s3://ai2-s2-lucas/s2orc_llm/2023_01_03/s2orc_clean/ \
    dst=... \
    cpu_count=1

"""

import gzip
import json
from collections import Counter
from contextlib import ExitStack

from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import pandas as pd
import springs as sp
from smashed.utils import io_utils

from .consts import COMMON_CUT, DATA_COLUMNS
from .lang_id import FasttextLangId, Cld2LangId
from .utils import (
    UnigramPerplexityPredictor,
    row_to_metadata,
    fix_missing_added,
    fix_missing_created,
    merge_text,
    s2orc_merge_headers,
    nfc_normalize
)
from .multiproc import make_runner, PbarUnit
from .cc_net import CCNet


def process_single(
    io_paths: Tuple[io_utils.MultiPath, io_utils.MultiPath],
    pbar_queue: Optional[Queue] = None,
    debug: bool = False,
):
    logger = sp.configure_logging(__name__, logging_level="WARNING", force_root_reattach=True)

    upp = UnigramPerplexityPredictor()
    ft_lang = FasttextLangId()
    cld2_lang = Cld2LangId()
    cc_net = CCNet()
    src, dst = io_paths
    dst.path += ".gz"

    with io_utils.open_file_for_read(src, "rb", logger=logger) as f, NamedTemporaryFile("wb") as tmp:
        tmp.write(f.read())
        tmp.flush()
        df = pd.read_parquet(tmp.name)

    # for debugging purposes, only take first 1000 rows
    if debug:
        df = df.head(1000)

    # filter all rows that don't have a "all_paragraphs" column
    df = df[df["all_paragraphs"].notna()]

    df["title"] = df["title"].apply(nfc_normalize)
    df["abstract"] = df["abstract"].apply(nfc_normalize)

    df["filtered_paragraphs"] = df["all_paragraphs"].apply(nfc_normalize).apply(s2orc_merge_headers)
    df.drop(columns=["all_paragraphs"], inplace=True)

    # assign version v0 and s2 as the source
    df["version"] = "v4"
    df["source"] = "s2"

    # fix missing added column
    df = df.apply(fix_missing_added, axis=1)

    # fix missing created column
    df = df.apply(fix_missing_created, axis=1)

    # spec requires id to be a string
    df["id"] = df["id"].astype(str)

    # if `fields_of_study` is not a list, then set it to an empty list
    df["fields_of_study"] = df["fields_of_study"].apply(lambda x: x if isinstance(x, list) else [])

    # create initial text by concatenating title and abstract and
    # all paragraphs
    df["text"] = df.apply(merge_text, axis=1)

    # create new column that is the result of the function
    # cld3.get_language(text) applied to the text column
    df["fstt_language_paragraphs"] = df["filtered_paragraphs"].apply(ft_lang.get_language)
    df["cld2_language_paragraphs"] = df["filtered_paragraphs"].apply(cld2_lang.get_language)

    # calculate the perplexity of each paragraph
    df["upp_perplexity_paragraphs"] = df["filtered_paragraphs"].apply(lambda x: [upp.predict(para) for para in x])
    df["ccnet_perplexity_paragraphs"] = df["filtered_paragraphs"].apply(lambda x: cc_net(x))

    # zip the language, perplexity, and filtered paragraphs columns together
    df["paragraphs"] = df.apply(
        lambda x: list(
            {
                "fasttext_language": fl,
                "cld2_language": cl,
                "upp_perplexity": up,
                "ccnet_perplexity": cp,
                "text": pp
            } for fl, cl, up, cp, pp in zip(
                x["fstt_language_paragraphs"],
                x["cld2_language_paragraphs"],
                x["upp_perplexity_paragraphs"],
                x["ccnet_perplexity_paragraphs"],
                x["filtered_paragraphs"]
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
    to_drop = [
        "fstt_language_paragraphs",
        "cld2_language_paragraphs",
        "upp_perplexity_paragraphs",
        "ccnet_perplexity_paragraphs",
        "filtered_paragraphs",
    ]
    df.drop(columns=to_drop, inplace=True)

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
        pbar_queue.put(PbarUnit.new(files=1, docs=int(len(df)), tokens=int(cnt)))

    del df


if __name__ == "__main__":
    CCNet().prefetch()
    make_runner(process_single)()
