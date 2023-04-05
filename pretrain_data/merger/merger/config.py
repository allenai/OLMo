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


class Stream(BaseModel):
    name: str
    format: Optional[str]
    documents: Documents
    attributes: Optional[Attributes]
    sampler: Optional[Sampler]
    filterer: Optional[Filterer]

    @property
    def formatter_fn(self):
        if not self.format or self.format == "ai2":
            return None
        raise Exception(f"Unknown format: {self.format}")


class Output(BaseModel):
    path: str
    max_file_size: str

    @property
    def max_file_size_in_bytes(self):
        if self.max_file_size.endswith("M"):
            return int(self.max_file_size[:-1]) * 1024**2
        elif self.max_file_size.endswith("G"):
            return int(self.max_file_size[:-1]) * 1024**3


class Config(BaseModel):
    streams: List[Stream]
    output: Output
    processes: Optional[int]
