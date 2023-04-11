from typing import List, Optional

from pydantic import BaseModel


class Documents(BaseModel):
    root: str
    include: List[str]


class Attributes(BaseModel):
    root: str
    include: List[str]


class Sampler(BaseModel):
    seed: int
    rate: float


class Filterer(BaseModel):
    include: Optional[List[str]]
    exclude: Optional[List[str]]


class Output(BaseModel):
    path: str
    max_shard_size: str

    @property
    def max_shard_size_in_bytes(self):
        if self.max_shard_size.endswith("M"):
            return int(self.max_shard_size[:-1]) * 1024**2
        elif self.max_shard_size.endswith("G"):
            return int(self.max_shard_size[:-1]) * 1024**3


class Stream(BaseModel):
    name: str
    format: Optional[str]
    documents: List[Documents]
    attributes: Optional[Attributes]
    sampler: Optional[Sampler]
    filterer: Optional[Filterer]
    output: Output

    @property
    def formatter_fn(self):
        if not self.format or self.format == "ai2":
            return None
        raise Exception(f"Unknown format: {self.format}")


class Config(BaseModel):
    streams: List[Stream]
    processes: Optional[int]
