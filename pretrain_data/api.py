"""

Run this to check if your data fields are appropriate

@kylel

"""

from typing import List, Dict, Iterable, Optional

import argparse
import logging
import sys
import gzip
from contextlib import ExitStack
import json

from smashed.utils.io_utils import recursively_list_files, MultiPath, open_file_for_read

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('_pretrain_data_api.log'),
    ])

SOURCE_TO_AVAILABLE_VERSIONS = {
    'common-crawl': ['v0'],
    'reddit': [],
    's2': ['v2_hard_dedup'],
    'stack-dedup': ['raw'],
    'wikipedia': []
}

SOURCE_TO_PATH = {
    'common-crawl': f"s3://ai2-llm/pretraining-data/sources/common-crawl/{SOURCE_TO_AVAILABLE_VERSIONS['common-crawl']}/documents/mined_split/*/*/*.json.gz",
    'reddit': None,
    's2': f"s3://ai2-llm/pretraining-data/sources/s2/{SOURCE_TO_AVAILABLE_VERSIONS['s2']}/*/*/*.gz",
    'stack-dedup': f"s3://ai2-llm/pretraining-data/sources/stack-dedup/{SOURCE_TO_AVAILABLE_VERSIONS['stack-dedup']}/*/*.jsonl.gz",
    'wikipedia': None
}


class Example:
    REQUIRED_FIELDS = ['source', 'id', 'text', 'added', 'created', 'metadata']

    def __init__(self, source: str, id: str, text: str, added: str, created: str, metadata: Dict):
        self.source = source
        self.id = id
        self.text = text
        self.added = added
        self.created = created
        self.metadata = metadata
        self._global_id = f'{self.source}::{self.id}'
        self._s3_filepath: Optional[str] = None
        self._s3_fileline: Optional[int] = None

    @property
    def global_id(self) -> str:
        return self._global_id

    @property
    def s3_filepath(self) -> Optional[str]:
        return self._s3_filepath

    @s3_filepath.setter
    def s3_filepath(self, s3_filepath: str):
        self._s3_filepath = s3_filepath

    @property
    def s3_fileline(self) -> Optional[str]:
        return self._s3_fileline

    @s3_fileline.setter
    def s3_fileline(self, s3_fileline: str):
        self._s3_fileline = s3_fileline

    @classmethod
    def from_json(cls, example_json: Dict) -> 'Example':
        example = Example(
            source=example_json['source'],
            id=example_json['id'],
            text=example_json['text'],
            added=example_json['added'],
            created=example_json['created'],
            metadata=example_json['metadata']
        )
        extra_keys = [key for key in example_json.keys() if key not in cls.REQUIRED_FIELDS]
        if extra_keys:
            logging.warning(msg=f"Extra keys found in Example JSON: {extra_keys}")
        return example

    def to_json(self) -> Dict:
        return {
            'source': self.source,
            'id': self.id,
            'text': self.text,
            'added': self.added,
            'created': self.created,
            'metadata': self.metadata
        }


class Dataset:
    def __init__(self, source: str, version: str, attributes: List[str]):
        logging.info(f'Creating Dataset from S3: source={source} version={version}')
        if source not in SOURCE_TO_PATH:
            raise FileNotFoundError(f'{source} not one of the available sources')
        if version not in SOURCE_TO_AVAILABLE_VERSIONS[source]:
            raise FileNotFoundError(f'{version} does not exist for {source}')

        self.source = source
        self.version = version
        self.attributes = attributes

        path_str = SOURCE_TO_PATH[source]
        i = path_str.index('*')
        dir_path = path_str[:i]
        self.s3_dirpath = dir_path

        self._s3_filepaths = recursively_list_files(path=MultiPath.parse(path=dir_path))

    @property
    def examples(self) -> Iterable[Example]:
        for s3_filepath in self.s3_filepaths:
            for example in self._read_examples_from_file(s3_filepath=s3_filepath):
                yield example

    @property
    def s3_filepaths(self) -> Iterable[str]:
        for path in self._s3_filepaths:
            yield path.as_str

    def _read_examples_from_file(self, s3_filepath: str) -> Iterable[Example]:
        with ExitStack() as stack:
            in_f = stack.enter_context(open_file_for_read(s3_filepath, "rb"))
            in_stream = stack.enter_context(gzip.open(in_f, "rt"))

            for i, raw in enumerate(in_stream):
                example_dict = json.loads(raw)
                example = Example.from_json(example_json=example_dict)
                example.s3_filepath = s3_filepath
                example.s3_fileline = i
                yield example

    def verify_all_examples(self):
        seen = set()
        for example in self.examples:
            if example.global_id in seen:
                raise ValueError(f"{example.global_id} already exists in this dataset")
            seen.add(example.global_id)

    def verify_one_file(self, s3_filepath: str):
        seen = set()
        for example in self._read_examples_from_file(s3_filepath=s3_filepath):
            if example.global_id in seen:
                raise ValueError(f"{example.global_id} already exists in this dataset")
            seen.add(example.global_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--attributes", nargs='*', required=False)
    args = parser.parse_args()

    dataset = Dataset(source=args.source, version=args.version, attributes=args.attributes)
    logging.info(f'Found one dataset from source={args.source} version={args.version}')

    first_s3_filepath = next(dataset.s3_filepaths, None)
    if first_s3_filepath:
        logging.info(f'Inspecting first file at {first_s3_filepath}')

        dataset.verify_one_file(s3_filepath=first_s3_filepath)
        logging.info(f'Finished verifying format of file {first_s3_filepath}')
    else:
        raise FileNotFoundError(f'No files found for source={args.source} version={args.version}')
