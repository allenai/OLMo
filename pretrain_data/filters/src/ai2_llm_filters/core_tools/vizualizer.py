import argparse
import os
from contextlib import ExitStack

import msgspec
from smashed.utils.io_utils import (
    decompress_stream,
    recursively_list_files,
    stream_file_for_read,
)
from termcolor import colored

from .data_types import DocResult, InputSpec, OutputSpec


class Visualizer:
    BASE_S3_PREFIX = "s3://ai2-llm/pretraining-data/sources"

    def __init__(self, dataset: str, experiment: str, tagger: str, type: str):
        self.dataset = dataset
        self.experiment = experiment
        self.tagger = tagger
        self.type = type
        # self.number = number

    def visualize_single(self, path: str):
        prefix, doc_path = path.split("/documents/")

        attrs_decoder = msgspec.json.Decoder(OutputSpec)
        doc_decoder = msgspec.json.Decoder(InputSpec)

        with ExitStack() as stack:
            doc_file = stack.enter_context(stream_file_for_read(path, "rb"))
            doc_stream = stack.enter_context(decompress_stream(doc_file, "rt"))
            exp_path = f"{prefix}/attributes/{self.experiment}/{doc_path}"
            exp_file = stack.enter_context(stream_file_for_read(exp_path, "rb"))
            exp_stream = stack.enter_context(decompress_stream(exp_file, "rt"))

            i = 0
            short_path = f"...{path[:s]}" if (s := os.get_terminal_size().columns // 2) < len(path) else path
            file_header = colored(f"file:    {short_path}\n", color="magenta")

            for doc_line, exp_line in zip(doc_stream, exp_stream):
                # parse out data from the line
                input_doc = doc_decoder.decode(doc_line)
                input_exp = attrs_decoder.decode(exp_line)
                doc_result = DocResult.from_spec(input_doc, input_exp)

                example_header = colored(f"example: {i:,}\n", color="yellow")
                dt = doc_result.doc.text

                spans = sorted(
                    (s for s in doc_result.spans if s.tagger == self.tagger and s.type == self.type),
                    key=lambda s: s.start,
                )
                if not spans:
                    continue

                prev_start = 0
                text_fragments = []
                for span in spans:
                    text_fragments.append(colored(dt[prev_start : span.start].replace("\n", "\\n"), color="black"))
                    text_fragments.append(colored(dt[span.start : span.end].replace("\n", "\\n"), color="green"))
                    text_fragments.append(colored(f"{{{span.type}: {span.score}}}", color="red"))
                    prev_start = span.end
                text_fragments.append(colored(dt[prev_start:].replace("\n", "\\n"), color="black"))

                tagger_header = colored(f"tagger:  {self.tagger}\n", color="cyan")
                print("".join(text_fragments) + file_header + example_header + tagger_header + "\n")
                while True:
                    out = input("next? [l/f] ").lower().strip()
                    if out == "l":
                        i += 1
                        break
                    elif out == "f":
                        return
                    else:
                        print(f"invalid input: {out}; choose between next (l)ine or next (f)ile.")

    def __call__(self):
        try:
            source_prefix = f"{self.BASE_S3_PREFIX}/{self.dataset}/documents"
            for path in recursively_list_files(source_prefix):
                self.visualize_single(path)
        except KeyboardInterrupt:
            print("\nExiting... bye!")

    @classmethod
    def main(cls):
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "-d",
            "--dataset",
            required=True,
            help="Dataset to visualize",
        )
        ap.add_argument(
            "-e",
            "--experiment-name",
            required=True,
            help="Experiment name to visualize",
        )
        ap.add_argument(
            "-t",
            "--tagger",
            required=False,
            help="Tagger to visualize",
        )
        ap.add_argument(
            "-y",
            "--type",
            required=True,
            help="Type to visualize",
        )
        opts = ap.parse_args()
        cls(dataset=opts.dataset, experiment=opts.experiment_name, tagger=opts.tagger, type=opts.type)()
