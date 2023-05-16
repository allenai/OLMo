"""
Example of how to use parallel utility for processing gzipped jsonl files containing pretraining data.

Author: Luca Soldaini (@soldni)
"""

import gzip
import json
import re
from contextlib import ExitStack
from typing import Dict

import springs as sp
from smashed.utils.io_utils import open_file_for_read, open_file_for_write

from ai2_llm_filters.parallel import BaseParallelProcessor


def get_word_count(text: str) -> int:
    """Count the number of words in a string by splitting on whitespaces and punctuation marks."""
    return sum(1 for _ in re.split(r"([.,!?;:]|\s?)\s*", text))


class CountWords(BaseParallelProcessor):
    """This example filter counts the number of words in a set of gzipped jsonl files and seves it to a
    destination."""

    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue,  # queue must be the first argument, and it should be a positional-only argument
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        """We override this method to specify which units we want to keep track of in a progress bar.
        Specifically, we keep track of files and documents in this example. Their default value must be zero."""

        # we call the super method to increment the progress bar
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue,
    ):
        """Lets count the number of word! We will use the destination path to save the number of lines
        for each file."""

        docs_cnt = 0
        with ExitStack() as stack:
            # open each file for reading and writing. We use open_file_for_read to handle s3 paths and
            # download the file locally if needed, while gzip.open is used to
            # read and write gzipped files.
            in_file = stack.enter_context(open_file_for_read(source_path, "rb"))
            in_stream = stack.enter_context(gzip.open(in_file, "rt"))
            out_file = stack.enter_context(open_file_for_write(destination_path, "wb"))
            out_stream = stack.enter_context(gzip.open(out_file, "wt"))

            for raw in in_stream:
                row = json.loads(raw)
                output = {"id": row["id"], "count": get_word_count(row["text"])}
                out_stream.write(json.dumps(output) + "\n")  # pyright: ignore

                # increment the number of documents processed so far
                docs_cnt += 1

                if docs_cnt % 1000 == 0:
                    # update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(queue, documents=docs_cnt)
                    docs_cnt = 0

        # increment the files progress bar
        cls.increment_progressbar(queue, files=1, documents=docs_cnt)


@sp.dataclass
class Config:
    source_prefix: str = "s3://ai2-llm/pretraining-data/sources/wikibooks/v0/documents/lang=en"
    destination_prefix: str = "s3://ai2-s2-lucas/tmp/parallel-test/counts"
    metadata_prefix: str = "s3://ai2-s2-lucas/tmp/parallel-test/metadata"
    num_processes: int = 2
    seed: int = 42
    debug: bool = False
    ignore_existing: bool = False


@sp.cli(Config)
def main(config: Config):
    CountWords(**sp.to_dict(config))()  # pyright: ignore


if __name__ == "__main__":
    main()
