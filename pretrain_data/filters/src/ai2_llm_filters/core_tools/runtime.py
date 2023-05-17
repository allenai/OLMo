import gzip
import json
import multiprocessing
import tempfile
from contextlib import ExitStack
from queue import Queue
from typing import Dict, List

from smashed.utils.io_utils import open_file_for_read, open_file_for_write

from .parallel import BaseParallelProcessor
from .registry import TaggerRegistry
from .utils import make_variable_name


class TaggerProcessor(BaseParallelProcessor):
    BASE_S3_PREFIX = "s3://ai2-llm/pretraining-data/sources"

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

        # get names of taggers
        taggers_names = kwargs.get("taggers_names", None)
        if taggers_names is None:
            raise RuntimeError("Taggers not in kwargs, this is a bug! Please report it.")
        elif not isinstance(taggers_names, list) or not all(isinstance(t, str) for t in taggers_names):
            raise RuntimeError("Taggers are in the wrong format, this is a bug! Please report it.")
        taggers = {make_variable_name(t): TaggerRegistry.get(t)() for t in taggers_names}

        # get name of experiment
        experiment_name = kwargs.get("experiment_name", None)
        if experiment_name is None:
            raise RuntimeError("Experiment name not in kwargs, this is a bug! Please report it.")
        experiment_name = make_variable_name(experiment_name)

        # interval at which to update the progress bar; will double if it gets
        # too full
        update_interval = 1

        # running document count; gets reset every time we update the progress
        # bar
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
                assert isinstance(row, dict), f"Expected dict, got {type(row)}"

                # running the taggers and merging them flat
                attributes = {}
                for tagger_name, tagger in taggers.items():
                    for key_name, key_value in tagger.tag(row).items():
                        key_name = f"{experiment_name}__{tagger_name}__{make_variable_name(key_name)}"
                        attributes[key_name] = key_value
                    attributes.update(tagger.tag(row))

                # make output file
                output = {"source": row["source"], "id": row["id"], "attributes": attributes}

                # write the output to the output file
                out_stream.write(json.dumps(output) + "\n")  # pyright: ignore

                # increment the number of documents processed so far
                docs_cnt += 1

                if docs_cnt % update_interval == 0:
                    # update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(queue, documents=docs_cnt)
                    docs_cnt = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_interval *= 2

        # increment the files progress bar
        cls.increment_progressbar(queue, files=1, documents=docs_cnt)

    @classmethod
    def main(
        cls,
        dataset: str,
        taggers: List[str],
        num_processes: int = 1,
        debug: bool = False,
    ):
        assert len(taggers) > 0, "At least one tagger must be specified"

        source_prefix = f"{cls.BASE_S3_PREFIX}/{dataset}/documents"
        destination_prefix = f"{cls.BASE_S3_PREFIX}/{dataset}/attributes"

        with tempfile.TemporaryDirectory() as tempdir:
            msg = (
                "----- TaggerProcessor -----\n"
                f"source:       {source_prefix}\n"
                f"destination:  {destination_prefix}\n"
                f"scratch:      {tempdir}\n"
                f"taggers:      {' -> '.join(taggers)}\n"
                f"parallel:     {num_processes}\n"
                "---------------------------\n"
            )
            print(msg)

            parallel_compute = cls(
                source_prefix=source_prefix,
                destination_prefix=destination_prefix,
                metadata_prefix=tempdir,
                num_processes=num_processes,
                ignore_existing=True,
                debug=debug,
            )
            parallel_compute(taggers_names=taggers)
