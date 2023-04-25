import datetime
import gzip
from hashlib import sha1
import json
import re
from typing import Any, Dict, Optional, Tuple
import dateparser
import springs as sp
from smashed.utils.io_utils import (
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)
from contextlib import ExitStack

import tqdm


PROJECT_GUTENBERG_START_DATE = 'December, 1971'
PROJECT_GUTENBERG_HEADERS_SEPARATORS = {
    '***START OF THE PROJECT GUTENBERG EBOOK',
    '*** START OF THIS PROJECT GUTENBERG EBOOK',
    '*** START OF THE PROJECT GUTENBERG EBOOK'
}
PROJECT_GUTENBERG_FOOTERS_SEPARATORS = {
    '***END OF THE PROJECT GUTENBERG EBOOK',
    '*** END OF THIS PROJECT GUTENBERG EBOOK',
    '*** END OF THE PROJECT GUTENBERG EBOOK'
}


def split_header_and_body(content: str) -> Tuple[str, str]:
    for separator in PROJECT_GUTENBERG_HEADERS_SEPARATORS:
        if separator in content:
            header, body = content.split(separator, 1)
            break
    else:
        raise ValueError('No header separator found in content')

    for separator in PROJECT_GUTENBERG_FOOTERS_SEPARATORS:
        if separator in body:
            body, footer = body.split(separator, 1)
            break
    else:
        raise ValueError('No footer separator found in content')

    return header, body.strip()


def count_words(text: str) -> int:
    # length is calculated using a regex that splits on whitespace
    return re.sub(r"\s+", " ", text).count(" ")


def format_timestamp(ts: Optional[datetime.datetime] = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()

    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


@sp.dataclass
class GutenbergProcessorConfig:
    src: str = 's3://ai2-llm/pretraining-data/sources/gutenberg/raw/files'
    dst: str = 's3://ai2-llm/pretraining-data/sources/gutenberg/v0/documents'


@sp.cli(GutenbergProcessorConfig)
def process(config: GutenbergProcessorConfig) -> None:

    paths = list(recursively_list_files(config.src))

    out_cnt = 0
    with ExitStack() as in_stack, ExitStack() as out_stack:
        in_progress = in_stack.enter_context(tqdm.tqdm(total=len(paths), desc='Reading files', unit='f'))
        out_progress = in_stack.enter_context(tqdm.tqdm(desc='Writing files', unit='f'))

        out_file = out_stack.enter_context(open_file_for_write(f'{config.dst}/{out_cnt:03d}.jsonl.gz', 'wb'))
        out_stream = out_stack.enter_context(gzip.open(out_file, mode='wt'))

        for fn in paths:

            content = None
            for encoding in ['utf-8', 'latin-1']:
                try:
                    with open_file_for_read(fn, open_kwargs={'encoding': encoding}) as in_file:
                        content = in_file.read()
                        break
                except Exception:
                    ...

            if content is None:
                print('[WARNING] Skipping file with unknown encoding:', fn)
                continue

            in_progress.update(1)

            if out_stream.tell() > 200_000_000:
                out_stream.close()
                out_file.close()
                out_cnt += 1
                out_file = out_stack.enter_context(open_file_for_write(f'{config.dst}/{out_cnt:03d}.jsonl.gz', 'wb'))
                out_stream = out_stack.enter_context(
                    gzip.open(out_file, mode='wt'))
                out_progress.update(1)

            try:
                header, body = split_header_and_body(content)
            except ValueError:
                # a couple of files don't have the header, so we skip them
                print('[WARNING] Skipping file without header:', fn)
                continue

            # we extract metadata from the header
            metadata: Dict[str, Any] = {}
            for line in header.splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    metadata[key] = value.strip()

            metadata['file_name'] = fn.split('/')[-1]
            metadata['length'] = count_words(body)

            created = dateparser.parse(
                # `[` might be present in the date, so we split on it (it is usually for notes)
                metadata.get('release_date', PROJECT_GUTENBERG_START_DATE).split('[', 1)[0]
            )

            document = {
                'id': sha1(body.encode('utf-8')).hexdigest(),
                'text': body,
                'created': format_timestamp(created),
                'added': format_timestamp(),
                'source': 'gutenberg',
                'version': 'v0',
                'metadata': metadata,
            }

            out_stream.write(json.dumps(document) + '\n')   # type: ignore


if __name__ == '__main__':
    process()
