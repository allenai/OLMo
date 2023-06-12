import argparse
import json
import os
from contextlib import ExitStack
from typing import Dict, List, Optional

import msgspec
import yaml
from smashed.utils.io_utils import (
    decompress_stream,
    recursively_list_files,
    stream_file_for_read,
)
from termcolor import colored

from .data_types import DocResult, InputSpec, OutputSpec


class Visualizer:
    BASE_S3_PREFIX = "s3://ai2-llm/pretraining-data/sources"

    def __init__(
        self,
        dataset: str,
        experiment: Optional[str] = None,
        tagger: Optional[str] = None,
        type: Optional[str] = None,
    ):
        self.dataset = dataset
        self.experiment = experiment
        self.tagger = tagger
        self.type = type

    def list_tags(self, path: str):
        prefix, doc_path = path.split("/documents/")

        attrs_decoder = msgspec.json.Decoder(OutputSpec)
        doc_decoder = msgspec.json.Decoder(InputSpec)

        with ExitStack() as stack:
            doc_file = stack.enter_context(stream_file_for_read(path, "rb"))
            doc_stream = stack.enter_context(decompress_stream(doc_file, "rt"))
            exp_path = f"{prefix}/attributes/{self.experiment}/{doc_path}"
            exp_file = stack.enter_context(stream_file_for_read(exp_path, "rb"))
            exp_stream = stack.enter_context(decompress_stream(exp_file, "rt"))

            tags: Dict[str, List[str]] = {}
            for doc_line, exp_line in zip(doc_stream, exp_stream):
                # parse out data from the line
                input_doc = doc_decoder.decode(doc_line)
                input_exp = attrs_decoder.decode(exp_line)
                doc_result = DocResult.from_spec(input_doc, input_exp)

                for span in doc_result.spans:
                    tags.setdefault(str(span.tagger), []).append(span.type)

                break

            print(colored(f"from {self.short_path(path)}:", color="yellow"))
            for tagger, types in sorted(tags.items()):
                print(colored(f"{tagger}:", color="magenta"))
                for type in sorted(set(types)):
                    print(colored(f"  {type}", color="cyan"))
                print()

    def short_path(self, path: str, slack: int = 20) -> str:
        return f"...{path[-s:]}" if (s := round(os.get_terminal_size().columns - slack)) < len(path) else path

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
            file_header = colored(f"file:    {self.short_path(path)}\n", color="magenta")

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
                print("".join(text_fragments) + "\n" + file_header + example_header + tagger_header + "\n")
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
                # just list tags if no tagger or type is specified
                if self.tagger is None or self.type is None:
                    self.list_tags(path)
                    return

                # visualize the specified tagger and type
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
            type=str,
            default=None,
            help="Tagger to visualize",
        )
        ap.add_argument(
            "-y",
            "--type",
            type=str,
            default=None,
            help="Type to visualize",
        )
        opts = ap.parse_args()
        cls(dataset=opts.dataset, experiment=opts.experiment_name, tagger=opts.tagger, type=opts.type)()


class RawPreviewer:
    BASE_S3_PREFIX = "s3://ai2-llm/pretraining-data/sources"

    def __init__(self, dataset: str, type: str, file: str, pretty: bool = False, experiment: Optional[str] = None):
        self.dataset = dataset
        self.experiment = experiment
        self.type = type
        self.file = file
        self.pretty = pretty

        assert type == "documents" or experiment is not None, "Must specify experiment for attributes"

    def preview_file(self):
        if self.type == "documents":
            path = f"{self.BASE_S3_PREFIX}/{self.dataset}/documents/{self.file}"
        else:
            path = f"{self.BASE_S3_PREFIX}/{self.dataset}/attributes/{self.experiment}/{self.file}"

        with ExitStack() as stack:
            file = stack.enter_context(stream_file_for_read(path, "rb"))
            stream = stack.enter_context(decompress_stream(file, "rt"))

            list_colors = ["red", "green", "blue", "magenta", "cyan"]

            for line in stream:
                row = json.loads(line)
                if self.pretty:
                    out = yaml.dump(row, width=float("inf"))
                    for ln in out.split("\n"):
                        if not ln.startswith("  "):
                            key, *rest = ln.split(":")
                            rest = (":" + ":".join(rest) if rest else "").strip()
                            print(colored(key, color=list_colors[0]) + rest)
                            list_colors = list_colors[1:] + list_colors[:1]
                        else:
                            print(ln)
                else:
                    print(row)
                input(colored("\n[press enter for next line]", color="yellow"))

    def list_files(self):
        prefix = f"{self.BASE_S3_PREFIX}/{self.dataset}/documents"
        for path in recursively_list_files(prefix):
            print(path[len(prefix) + 1 :])

    def __call__(self):
        try:
            if self.file is not None:
                self.preview_file()
            else:
                self.list_files()
        except KeyboardInterrupt:
            print("\nExiting... bye!")

    @classmethod
    def main(cls):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True, help="Dataset to preview, e.g. `wikipedia/v0`")
        ap.add_argument(
            "-t",
            "--type",
            choices=["documents", "attributes"],
            required=True,
            help="Type of data to preview; it can be either `documents` or `attributes`.",
        )
        ap.add_argument(
            "-e",
            "--experiment",
            default=None,
            help="Experiment to preview; this is only used for previewing `attributes` assigned by tagger.",
        )
        ap.add_argument(
            "-f",
            "--file",
            default=None,
            type=str,
            help="File to preview; if not sure which file to preview, skip this argument to list all files.",
        )
        ap.add_argument(
            "-p", "--pretty", action="store_true", help="Whether to use pretty print for previewing JSON lines."
        )
        opts = ap.parse_args()

        cls(dataset=opts.dataset, type=opts.type, file=opts.file, experiment=opts.experiment, pretty=opts.pretty)()
