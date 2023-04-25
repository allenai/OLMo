import datetime
import gzip
import json
import re
from contextlib import ExitStack
from hashlib import sha1
from typing import Any, Dict, Optional

import dateparser
import gutenbergpy.textget
import springs as sp
import tqdm
from smashed.utils.io_utils import (
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)

PROJECT_GUTENBERG_START_DATE = "December 1, 1971"


def count_words(text: str) -> int:
    # length is calculated using a regex that splits on whitespace
    return re.sub(r"\s+", " ", text).count(" ")


def format_timestamp(ts: Optional[datetime.datetime] = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()

    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


@sp.dataclass
class GutenbergProcessorConfig:
    src: str = "s3://ai2-llm/pretraining-data/sources/gutenberg/raw/json"
    dst: str = "s3://ai2-llm/pretraining-data/sources/gutenberg/v0/documents"


@sp.cli(GutenbergProcessorConfig)
def process(config: GutenbergProcessorConfig) -> None:
    paths = list(recursively_list_files(config.src))

    out_cnt = 0
    with ExitStack() as in_stack, ExitStack() as out_stack:
        in_progress = in_stack.enter_context(tqdm.tqdm(total=len(paths), desc="Reading files", unit="f"))
        out_progress = in_stack.enter_context(tqdm.tqdm(desc="Writing files", unit="f"))

        out_file = out_stack.enter_context(open_file_for_write(f"{config.dst}/{out_cnt:05d}.jsonl.gz", "wb"))
        out_stream = out_stack.enter_context(gzip.open(out_file, mode="wt"))

        for fn in paths:
            content = None

            with open_file_for_read(fn, "r") as in_file:
                content = json.loads(in_file.read())

            if content is None:
                print("[WARNING] Skipping file with unknown encoding:", fn)
                continue

            in_progress.update(1)

            if out_stream.tell() > 500_000_000:
                out_stack.pop_all().close()
                out_cnt += 1
                out_file = out_stack.enter_context(
                    open_file_for_write(f"{config.dst}/{out_cnt:05d}.jsonl.gz", "wb")
                )
                out_stream = out_stack.enter_context(gzip.open(out_file, mode="wt"))
                out_progress.update(1)

            body = gutenbergpy.textget.strip_headers(content["raw"].encode("utf-8")).decode("utf-8")

            metadata: Dict[str, Any] = {
                "rdf_available": "rdf" in content and len(content["rdf"]) > 0,
                "pipeline": content.get("pipeline", None),
            }

            gt_metadata_candidates = [
                e for e in content["rdf"] if e["@id"] == f'http://www.gutenberg.org/ebooks/{content["id"]}'
            ]
            if len(gt_metadata_candidates) == 0:
                metadata["gutenberg_metadata_available"] = False
            else:
                metadata.update(
                    {
                        k.replace("http://", ""): [v["@value"] for v in v if "@value" in v][0]
                        for k, v in gt_metadata_candidates[0].items()
                        if isinstance(v, list) and any("@value" in e for e in v)
                    }
                )
                metadata["gutenberg_metadata_available"] = True

            # find creation date
            issued_or_updated = (
                metadata.get("purl.org/dc/terms/issued", None)
                or metadata.get("purl.org/dc/terms/modified", None)
                or metadata.get("purl.org/dc/terms/date", None)
                or metadata.get("www.gutenberg.org/2009/pgterms/marc508", None)
            )

            metadata["issued_or_updated_available"] = False
            created = dateparser.parse(PROJECT_GUTENBERG_START_DATE)

            if isinstance(issued_or_updated, str):
                re_match = re.search(r"(\d{4})(-[0-1][0-9])?(-[0-3][0-9])?", issued_or_updated)
                if re_match:
                    created = dateparser.parse(re_match.group(0))
                    metadata["issued_or_updated_available"] = True

            # remove 4 newlines in a row for two
            body = re.sub(r"\n{4}", r"\n\n", body.strip())

            # get approximate length by counting spaces
            metadata["length"] = count_words(body)

            document = {
                "id": sha1(body.encode("utf-8")).hexdigest(),
                "text": body,
                "created": format_timestamp(created),
                "added": format_timestamp(),
                "source": "gutenberg",
                "version": "v0",
                "metadata": metadata,
            }

            out_stream.write(json.dumps(document) + "\n")  # type: ignore


if __name__ == "__main__":
    process()
