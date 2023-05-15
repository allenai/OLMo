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

    # assign version v0 and s2 as the source
    df["version"] = "v4"
    df["source"] = "s2"

    # fix missing added column
    df = df.apply(fix_missing_added, axis=1)

    # fix missing created column
    df = df.apply(fix_missing_created, axis=1)

    # spec requires id to be a string
    df["id"] = df["id"].astype(str)

    # normalize the text columns
    df["title"] = df["title"].apply(nfc_normalize)
    df["abstract"] = df["abstract"].apply(nfc_normalize)

    # create initial text by concatenating title and abstract
    df["text"] = df["title"] + "\n" + df["abstract"]

    # create new column that is the result of the function
    df["fstt_language_title"] = df["title"].apply(ft_lang.get_language)
    df["cld2_language_title"] = df["title"].apply(cld2_lang.get_language)
    df["fstt_language_abstract"] = df["abstract"].apply(ft_lang.get_language)
    df["cld2_language_abstract"] = df["abstract"].apply(cld2_lang.get_language)

    # calculate the perplexity of abstract and title
    df["upp_perplexity_title"] = df["title"].apply(upp.predict)
    df["ccnet_perplexity_title"] = df["title"].apply(lambda x: cc_net([x])[0])
    df["upp_perplexity_abstract"] = df["abstract"].apply(upp.predict)
    df["ccnet_perplexity_abstract"] = df["abstract"].apply(lambda x: cc_net([x])[0])

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

    # listify the sources
    df["sources"] = df["sources"].apply(lambda x: [] if x is None else x.tolist())

    # put everything that is not part of the data spec in metadata
    df["metadata"] = df.apply(row_to_metadata, axis=1)
    cnt = sum(df["title_count"] + df["abstract_count"])
    df = df.drop([c for c in df.columns if c not in DATA_COLUMNS], axis=1)

    with ExitStack() as stack:
        out_f = stack.enter_context(io_utils.open_file_for_write(dst, "wb"))
        out_stream = stack.enter_context(gzip.open(out_f, "wt"))
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            content = json.dumps(row_dict, default=str).strip()
            out_stream.write(content + "\n")  # type: ignore

    if pbar_queue is not None:
        pbar_queue.put(PbarUnit.new(files=1, docs=len(df), tokens=int(cnt)))

    del df


if __name__ == "__main__":
    CCNet().prefetch()
    make_runner(process_single)()
