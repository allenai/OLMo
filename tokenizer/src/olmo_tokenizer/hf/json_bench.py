import json
import time

import orjson
from msgspec.json import decode
from msgspec import Struct

import gzip
from smashed.utils.io_utils import open_file_for_read

path = '/Users/lucas/Downloads/c4-train.00930-of-01024.jsonl.gz'
cnt = 100_000

start = time.time()
with open_file_for_read(path, 'rb') as f:
    with gzip.open(f, 'rt') as stream:
        for i, line in enumerate(stream):
            out = json.loads(line)

            if i > cnt:
                break
end = time.time()
print(f'json: {end - start}')

start = time.time()
with open_file_for_read(path, 'rb') as f:
    with gzip.open(f, 'rt') as stream:
        for i, line in enumerate(stream):
            orjson.loads(line)

            if i > cnt:
                break
end = time.time()
print(f'orjson: {end - start}')



class Doc(Struct):
    text: str


start = time.time()
with open_file_for_read(path, 'rb') as f:
    with gzip.open(f, 'rt') as stream:
        for i, line in enumerate(stream):
            out = decode(line, type=Doc)
            import ipdb; ipdb.set_trace()

            if i > cnt:
                break
end = time.time()
print(f'msgpack: {end - start}')
