import re
from dataclasses import dataclass
from typing import Iterator, List

import boto3
from smart_open import open

s3 = boto3.client("s3")


@dataclass
class S3File:
    bucket: str
    key: str

    @classmethod
    def from_url(cls, url: str) -> "S3File":
        path_pattern = re.search("s3://([^/]*)/(.*)", url)
        bucket = path_pattern.group(1)  # type: ignore
        prefix = path_pattern.group(2)  # type: ignore
        if prefix.startswith("/"):
            prefix = prefix[1:]
        return S3File(bucket, prefix)

    @property
    def url(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def size(self) -> int:
        return s3.head_object(Bucket=self.bucket, Key=self.key)["ContentLength"]

    @property
    def children(self) -> List[str]:
        paginator = s3.get_paginator("list_objects_v2")
        return [
            f["Prefix"]
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.key, Delimiter="/")
            for f in page.get("CommonPrefixes", [])
        ]

    def read_lines(self) -> Iterator[str]:
        for line in open(self.url):
            yield line
