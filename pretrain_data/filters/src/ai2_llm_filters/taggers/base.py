import gzip
import json
import re
from abc import abstractmethod
from contextlib import ExitStack
from queue import Queue
from tempfile import tempdir
from typing import Dict, Generator, Iterable, List, Tuple, Type, Union

import springs as sp
from smashed.utils.io_utils import open_file_for_read, open_file_for_write

from ..parallel import BaseParallelProcessor


class BaseTagger:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def tag(self, text: str) -> dict:
        raise NotImplementedError


class TaggerRegistry:
    __taggers: dict

    @classmethod
    def taggers(cls) -> Generator[Tuple[str, Type[BaseTagger]], None, None]:
        yield from cls.__taggers.items()

    @classmethod
    def add(cls, tagger_cls: Type[BaseTagger]):
        cls.__taggers[tagger_cls.__name__] = tagger_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseTagger]:
        if name not in cls.__taggers:
            raise ValueError(
                f"Unknown tagger {name}; available taggers: " + ", ".join([tn for tn, _ in cls.taggers()])
            )
        return cls.__taggers[name]


@sp.dataclass
class TaggerConfig:
    dataset: str = sp.field(
        default=sp.MISSING, help="Name of the pretraining dataset to consider when applying filters, e.g. 's2/v4'"
    )
    name: str = sp.field(
        default=sp.MISSING,
        help="Name of the tagger to use, e.g. 'language-id-v1' ",
    )
    taggers: list = sp.field(
        default_factory=list,
        help="List of taggers to use, e.g. ['cld3']",
    )
    num_processes: int = sp.field(
        default=1,
        help="Number of processes to use for parallel processing",
    )


class TaggerProcessor(BaseParallelProcessor):
    BASE_S3_PREFIX = "s3://ai2-llm/pretrain_data"

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
        queue: "Queue",
        **kwargs,
    ):
        """Lets count the number of word! We will use the destination path to save the number of lines
        for each file."""

        config = TaggerConfig(**kwargs)
        taggers: List[BaseTagger] = [TaggerRegistry.get(t)() for t in config.taggers]

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

                # running the taggers
                output = {
                    "source": row["source"],
                    "id": row["id"],
                    "attributes": {tagger.__class__.__name__: tagger.tag(row["source"]) for tagger in taggers},
                }
                # write the output to the output file
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

    @classmethod
    def main(cls, config: TaggerConfig):
        parallel_compute = cls(
            source_prefix=f"{cls.BASE_S3_PREFIX}/{config.dataset}/documents",
            destination_prefix=f"{cls.BASE_S3_PREFIX}/{config.dataset}/attributes/{config.name}",
            metadata_prefix=f"{(tempdir or '/tmp').lstrip('/')}/{config.name}",
            num_processes=config.num_processes,
            ignore_existing=True,
        )
        parallel_compute(config=sp.to_dict(config))
