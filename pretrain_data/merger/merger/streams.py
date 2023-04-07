import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from zlib import error as ZlibError

from merger.config import Stream
from merger.merge import Merger
from merger.s3_util import S3File
from smart_open import open

log = logging.getLogger(__name__)


@dataclass
class DocumentInput:
    documents: S3File
    attributes: List[S3File]


def document_inputs(stream: Stream) -> List[DocumentInput]:
    root = S3File.from_url(stream.documents.root)
    if not root.key.endswith("/"):
        root.key = root.key + "/"
    keys = []
    for pattern in stream.documents.include:
        path = pattern.split("/")
        keys.extend(_matching_keys(root.bucket, root.key, path))
    keys.sort()
    return [
        DocumentInput(
            documents=S3File(bucket=root.bucket, key=k),
            attributes=[
                S3File(bucket=root.bucket, key=k.replace("/documents/", f"/attributes/{name}/"))
                for name in stream.attributes.include
            ]
            if stream.attributes
            else [],
        )
        for k in keys
    ]


def _matching_keys(bucket: str, prefix: str, path: List[str]) -> List[str]:
    if len(path) == 1:
        return [f"{prefix}{path[0]}"]
    elif path[0] == "*":
        f = []
        for child in S3File(bucket, prefix).children:
            f.extend(_matching_keys(bucket, child, path[1:]))
        return f
    else:
        return _matching_keys(bucket, f"{prefix}{path[0]}/", path[1:])


@dataclass
class Shard:
    inputs: List[DocumentInput]
    output: Path
    merger: Merger


def merged_size(input: DocumentInput) -> Tuple[DocumentInput, int]:
    return (input, input.documents.size + sum(a.size for a in input.attributes))


def split_into_shards(stream: Stream) -> List[Shard]:
    log.info(f"Computing shards for {stream.name}")
    inputs = document_inputs(stream)
    with mp.Pool(mp.cpu_count()) as p:
        inputs_with_size = p.map(merged_size, inputs)
    shards: List[Shard] = []
    sample_rate = stream.sampler.rate if stream.sampler else 1.0
    input, size = inputs_with_size[0]
    shard_inputs = [input]
    shard_size = int(sample_rate * size)
    for input, size in inputs_with_size[1:]:
        shard_size += int(sample_rate * size)
        if shard_size > stream.output.max_shard_size_in_bytes:
            shards.append(
                Shard(
                    inputs=shard_inputs,
                    output=Path(stream.output.path) / f"{stream.name}_{len(shards):04d}.json.gz",
                    merger=Merger(stream.sampler, stream.filterer, stream.formatter_fn),
                )
            )
            shard_size = 0
            shard_inputs = []
        shard_inputs.append(input)
    log.info(f"Splitting {len(inputs)} files for {stream.name} into {len(shards)} shards")
    return shards


def retry_process(shard: Shard) -> Tuple[Shard, Optional[Exception]]:
    max_retries = 3
    err = None
    for i in range(max_retries):
        err = process(shard)[1]
        if not err:
            return shard, None
        else:
            log.warning(f"Failed to process shard {shard.output} on attempt {i + 1}/{max_retries}: {err}")
    return shard, err


def process(shard: Shard) -> Tuple[Shard, Optional[Exception]]:
    try:
        if shard.output.exists():
            return shard, None
        log.info(f"Starting merge of {len(shard.inputs)} files to {shard.output}")
        shard.output.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = shard.output.parent / (shard.output.stem + ".tmp" + shard.output.suffix)
        tmp_file.unlink(missing_ok=True)
        with open(tmp_file, "wb") as w:
            for input in shard.inputs:
                log.info(
                    f"Merging documents from {input.documents.url} with {len(input.attributes)} attributes to {shard.output}"
                )
                try:
                    doc_line_iter = input.documents.read_lines()
                    attr_line_iters = [d.read_lines() for d in input.attributes]
                    for doc_line in doc_line_iter:
                        attr_lines = [next(i) for i in attr_line_iters]
                        output = shard.merger.merge(doc_line, attr_lines)
                        if output:
                            w.write(output)
                            w.write(b"\n")
                except (EOFError, ZlibError) as e:
                    log.warning(f"Bad input file {input.documents.url}. Skipping: {e}")
        tmp_file.rename(shard.output)
        return shard, None
    except Exception as e:
        return shard, e
