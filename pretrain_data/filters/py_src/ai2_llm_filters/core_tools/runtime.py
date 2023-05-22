import argparse
import multiprocessing
import tempfile
from contextlib import ExitStack
from queue import Queue
from typing import Dict, List

import msgspec
from smashed.utils.io_utils import (
    compress_stream,
    decompress_stream,
    open_file_for_write,
    stream_file_for_read,
)

from .data_types import InputSpec, OutputSpec
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

        # creating dedicated encoders/decoders speeds up the process
        encoder = msgspec.json.Encoder()
        decoder = msgspec.json.Decoder(InputSpec)

        with ExitStack() as stack:
            # open each file for reading and writing. We use open_file_for_read to handle s3 paths and
            # download the file locally if needed, while gzip.open is used to
            # read and write gzipped files.
            in_file = stack.enter_context(stream_file_for_read(source_path, "rb"))
            in_stream = stack.enter_context(decompress_stream(in_file, "rt"))
            out_file = stack.enter_context(open_file_for_write(destination_path, "wb"))
            out_stream = stack.enter_context(compress_stream(out_file, "wt"))

            for raw in in_stream:
                # row = json.loads(raw)
                row = decoder.decode(raw)

                # running the taggers and merging them flat
                attributes = {}
                for tagger_name, tagger in taggers.items():
                    for key_name, key_value in tagger.tag(row).items():
                        key_name = f"{experiment_name}__{tagger_name}__{make_variable_name(key_name)}"
                        attributes[key_name] = key_value

                # make output file
                output = OutputSpec(source=row.source, id=row.id, attributes=attributes)

                # write the output to the output file
                out_stream.write(encoder.encode(output).decode("utf-8") + "\n")  # pyright: ignore

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
    def main(cls):
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "-d",
            "--dataset",
            default=None,
            help=f"Dataset to process; this should be relative path from {TaggerProcessor.BASE_S3_PREFIX}.",
        )
        ap.add_argument(
            "-n",
            "--experiment-name",
            default=None,
            help=(
                "Name of for this sequence of taggers to be grouped under; "
                "it could be 'experiment_n' or a more descriptive name."
            ),
        )
        ap.add_argument(
            "-t",
            "--taggers",
            default=[],
            nargs="+",
            help="One or more taggers to run; use -l to list available taggers.",
        )
        ap.add_argument("-l", "--list-taggers", action="store_true", help="List available taggers.")
        ap.add_argument("-p", "--parallel", type=int, default=1, help="Number of parallel processes to use.")
        ap.add_argument(
            "-u", "--debug", action="store_true", help="Run in debug mode; parallelism will be disabled."
        )
        ap.add_argument(
            "--base-s3-prefix",
            default=cls.BASE_S3_PREFIX,
            help=f"Base S3 prefix to use; defaults to {cls.BASE_S3_PREFIX}.",
        )
        opts = ap.parse_args()

        if opts.list_taggers:
            print("Available taggers:")
            for tagger_name, tagger_cls in TaggerRegistry.taggers():
                print(f"  {tagger_name} ({tagger_cls.__name__})")
            return

        assert opts.dataset is not None, "Dataset must be specified."
        assert opts.experiment_name is not None, "Experiment name must be specified."
        assert len(opts.taggers) > 0, "At least one tagger must be specified."

        source_prefix = f"{cls.BASE_S3_PREFIX}/{opts.dataset}/documents"
        destination_prefix = f"{cls.BASE_S3_PREFIX}/{opts.dataset}/attributes/{opts.experiment_name}"

        with tempfile.TemporaryDirectory() as tempdir:
            msg = (
                "----- TaggerProcessor -----\n"
                f"source:       {source_prefix}\n"
                f"destination:  {destination_prefix}\n"
                f"scratch:      {tempdir}\n"
                f"taggers:      {', '.join(opts.taggers)}\n"
                f"parallel:     {opts.parallel}\n"
                "---------------------------\n"
            )
            print(msg)

            # override base s3 prefix
            cls.BASE_S3_PREFIX = opts.base_s3_prefix

            parallel_compute = cls(
                source_prefix=source_prefix,
                destination_prefix=destination_prefix,
                metadata_prefix=tempdir,
                num_processes=opts.parallel,
                ignore_existing=True,
                debug=opts.debug,
            )
            parallel_compute(taggers_names=opts.taggers, experiment_name=opts.experiment_name)
