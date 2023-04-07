import json
import random
from typing import Callable, Dict, List, Optional

import orjson
from jsonpath_ng.ext import parse
from merger.config import Filterer, Sampler


class Merger:
    def __init__(
        self, sampler: Optional[Sampler], filterer: Optional[Filterer], formatter: Optional[Callable[[Dict], Dict]]
    ):
        self.sampler = sampler
        self.rand = None
        self.formatter = formatter
        if sampler and sampler.rate < 1.0:
            self.rand = random.Random(sampler.seed)

        self.filterer = filterer
        self.include_patterns = []
        self.exclude_patterns = []
        if filterer:
            self.include_patterns = [parse(p) for p in (filterer.include or [])]
            self.exclude_patterns = [parse(p) for p in (filterer.exclude or [])]

    def merge(self, doc_line: str, attr_lines: List[str]) -> Optional[bytes]:
        if self.rand and self.sampler and self.rand.random() >= self.sampler.rate:
            return None
        if self.filterer is None and not attr_lines:
            # Nothing to merge, nothing to filter. Don't need to deserialize
            return doc_line.strip().encode("utf-8")
        doc = orjson.loads(doc_line)
        if self.formatter:
            doc = self.formatter(doc)
        attrs = {}
        for line in attr_lines:
            attrs.update(json.loads(line))
        if attrs:
            doc["attributes"] = attrs
        # If "include_patterns" is empty, then include everything, otherwise include only matches
        matches_include = (not self.include_patterns) or next(
            (p for p in self.include_patterns if p.find([doc])), None  # type: ignore
        )
        matches_exclude = self.exclude_patterns and next((p for p in self.exclude_patterns if p.find([doc])), None)  # type: ignore

        if (not matches_include) or matches_exclude:
            return None
        return orjson.dumps(doc)
