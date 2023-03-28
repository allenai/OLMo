import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from merger import config
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


def split_into_shards(stream: Stream, inputs: List[DocumentInput], output: config.Output) -> List[Shard]:
    # Estimate the size of an input file
    import random

    sample = random.sample(inputs, 10) if len(inputs) > 10 else inputs
    typical_size = sum(d.documents.size for d in sample) // len(sample)
    if stream.sampler:
        typical_size = int(typical_size * stream.sampler.rate)
    files_per_shard = output.max_file_size_in_bytes // typical_size
    shards = [
        Shard(
            inputs=inputs[i * files_per_shard : (i + 1) * files_per_shard],
            output=Path(output.path) / f"{stream.name}_{i:04d}.json.gz",
            merger=Merger(stream.sampler, stream.filterer, stream.formatter_fn),
        )
        for i in range(0, 1 + len(inputs) // files_per_shard)
    ]
    return shards


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
                doc_line_iter = input.documents.read_lines()
                attr_line_iters = [d.read_lines() for d in input.attributes]
                for doc_line in doc_line_iter:
                    attr_lines = [next(i) for i in attr_line_iters]
                    output = shard.merger.merge(doc_line, attr_lines)
                    if output:
                        w.write(output)
                        w.write(b"\n")
        tmp_file.rename(shard.output)
        log.info(f"Finished writing {shard.output}")
        return shard, None
    except Exception as e:
        return shard, e
