"""

Run this to check if your data fields are appropriate

@kylel

"""

from typing import List, Dict, Tuple

import os
import boto3
import botocore.exceptions
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('_pretrain_data_api.log'),
    ])

SOURCE_TO_LATEST_VERSION = {
    'common-crawl': 'v0',
    'reddit': None,
    's2': 'v2_hard_dedup',
    'stack-dedup': 'raw',
    'wikipedia': None
}

SOURCE_TO_PATHS = {
    'common-crawl': [
        f"pretraining-data/sources/common-crawl/{SOURCE_TO_LATEST_VERSION['common-crawl']}/documents/mined/*/*.json.gz",
        f"pretraining-data/sources/common-crawl/{SOURCE_TO_LATEST_VERSION['common-crawl']}/documents/mined_split/*/*/*.json.gz"
    ],
    'reddit': [],
    's2': [
        f"pretraining-data/sources/s2/{SOURCE_TO_LATEST_VERSION['s2']}/dataset=s2ag/split=train/*.gz",
        f"pretraining-data/sources/s2/{SOURCE_TO_LATEST_VERSION['s2']}/dataset=s2ag/split=valid/*.gz",
        f"pretraining-data/sources/s2/{SOURCE_TO_LATEST_VERSION['s2']}/dataset=s2orc/split=train/*.gz",
        f"pretraining-data/sources/s2/{SOURCE_TO_LATEST_VERSION['s2']}/dataset=s2ag/split=valid/*.gz"
    ],
    'stack-dedup': [
        f"pretraining-data/sources/stack-dedup/{SOURCE_TO_LATEST_VERSION['stack-dedup']}/*/*.jsonl.gz"
    ],
    'wikipedia': []
}


class S3File:
    s3 = boto3.resource('s3')

    def __init__(self, bucket: str, path: str):
        bucket = bucket.replace('s3://', '') if bucket.startswith('s3://') else bucket
        self.bucket = self.s3.Bucket(bucket)
        self.path = path
        try:
            self._object = self.s3.Object(bucket, path)
            self._object.load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                raise FileNotFoundError(f'Missing {self.path} in bucket {self.bucket.name}')
            else:
                raise Exception(f'Unknown exception for {self.path}')

    @property
    def url(self) -> str:
        return os.path.join(f's3://{self.bucket.name}', self.path)

    @property
    def size(self) -> int:
        # bytes
        return self._object.content_length

    def get(self, target_path: str, is_overwrite: bool = False) -> str:
        if os.path.exists(target_path) and not is_overwrite:
            raise FileExistsError(f'{target_path} already exists. Try `is_overwrite=True`')
        self.bucket.download_file(self.path, target_path)
        if os.path.exists(target_path):
            return target_path
        else:
            raise FileNotFoundError(f'Faild to download. Nothing at {target_path}')

    @classmethod
    def glob_to_files(cls, bucket: str, glob_path: str) -> List['S3File']:
        bucket = bucket.replace('s3://', '') if bucket.startswith('s3://') else bucket
        bucket = cls.s3.Bucket(bucket)

        # figure out top-level path to query
        path_chunks = glob_path.split('/')
        prefix = ''
        for i, path_chunk in enumerate(path_chunks):
            is_need_expand = path_chunk.startswith('*')
            if is_need_expand:
                break
            else:
                prefix = os.path.join(prefix, path_chunk)

        logging.info(f"Querying for S3 objects at {bucket.name} {prefix}")
        s3_objs = bucket.objects.filter(Prefix=prefix)
        s3_files = [
            S3File(bucket=bucket.name, path=obj.key)
            for obj in s3_objs if obj.key.endswith('.gz')
        ]
        logging.info(f"Found {len(s3_files)} S3 files at {bucket.name} {prefix}")
        return s3_files


class Example:
    def __init__(self, source: str, id: str, text: str, added: str, created: str, metadata: Dict):
        self.source = source
        self.id = id
        self.text = text
        self.added = added
        self.created = created
        self.metadata = metadata
        self._global_id = f'{self.source}::{self.id}'

    @property
    def global_id(self) -> str:
        return self._global_id

    @classmethod
    def from_json(cls, example_json: Dict) -> 'Example':
        example = Example(**example_json)
        return example


class Dataset:
    def __init__(self, examples: List[Example]):
        self.examples = examples
        self.verify_unique_source_id()

    def verify_unique_source_id(self):
        seen = set()
        for example in self.examples:
            if example.global_id in seen:
                raise ValueError(f"{example.global_id} already exists in this dataset")
            seen.add(example.global_id)

    @classmethod
    def from_s3(cls, bucket: str, source: str, version: str, attributes: List[str]) -> 'Dataset':

        logging.info(f'Creating Dataset from S3 at {bucket}: source={source} version={version}')

        if source not in SOURCE_TO_PATHS:
            raise FileNotFoundError(f'{source} not one of the available sources')

        s3_files = []
        for path in SOURCE_TO_PATHS[source]:
            s3_files.extend(S3File.glob_to_files(bucket=bucket, glob_path=path))

        import pdb;
        pdb.set_trace()


if __name__ == '__main__':
    dataset = Dataset.from_s3(bucket='ai2-llm',
                              source='stack-dedup',
                              version=SOURCE_TO_LATEST_VERSION['stack-dedup'],
                              attributes=[])
