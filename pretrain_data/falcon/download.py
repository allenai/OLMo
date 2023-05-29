import datetime
import json
import multiprocessing
import os
from argparse import ArgumentParser
from contextlib import ExitStack
from hashlib import md5
from queue import Queue
from typing import Dict, List, Tuple, Union

from ai2_llm_filters.core_tools.parallel import BaseParallelProcessor
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from smashed.utils.io_utils import MultiPath, compress_stream, open_file_for_write

NUM_EXAMPLES = 968_000_015
NUM_BYTES = 2_766_953_721_769
DOWNLOAD_SIZE = 466_888_198_663
N_SHARDS = 5_534
PARTITIONS = 500


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class FalconDownloader(BaseParallelProcessor):
    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
        hf_access_token: str,
    ):
        disable_progress_bar()

        dataset_name, shard_id_str = source_path.rsplit("/", 1)
        dataset = load_dataset(dataset_name, split="train", streaming=True, use_auth_token=hf_access_token)
        added = datetime.datetime.now()

        shard_id = int(shard_id_str)
        num_shards = PARTITIONS
        num_examples = int(NUM_EXAMPLES)
        shard_start = round(shard_id * num_examples / num_shards)
        shard_end = round((shard_id + 1) * num_examples / num_shards)

        if shard_start > 0:
            dataset = dataset.skip(shard_start)  # pyright: ignore

        doc_cnt = 0

        with ExitStack() as stack:
            dst_file = stack.enter_context(open_file_for_write(destination_path, "wb"))
            dst_stream = stack.enter_context(compress_stream(dst_file, "wt"))

            for i, row in enumerate(dataset):
                if (i := i + shard_start) >= shard_end:
                    break

                data = {
                    "text": row["content"],
                    "id": md5(row["url"].encode("utf-8")).hexdigest(),
                    "version": "v0",
                    "source": dataset_name.strip("/"),
                    "added": convert_timestamp(added),
                    "created": convert_timestamp(row["timestamp"]),
                    "metadata": {
                        "url": row["url"],
                        "dump": row["dump"],
                        "segment": row["segment"],
                        "image_urls": row["image_urls"],
                        "split": "train",
                        "pos": i,
                    },
                }
                dst_stream.write(json.dumps(data) + "\n")
                doc_cnt += 1
                if doc_cnt >= 1000:
                    cls.increment_progressbar(queue, documents=doc_cnt)
                    doc_cnt = 0

        # increment the files progress bar
        cls.increment_progressbar(queue, files=1, documents=doc_cnt)
        # print(f"Finished processing {shard_id:,} to {destination_path} ({shard_start:,} to {shard_end:,})\n\n")

    @classmethod
    def increment_progressbar(
        cls,
        queue: "Queue[Union[None, Tuple[int, ...]]]",
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        """Increment the progress bar by putting a tuple in the queue.

        When subclassing, we recommend defining which units to keep track of in the progress bar by
        defining keyword arguments. Then you can call the base class via `super()` and pass the keyword.
        Example:

        ```python
        class MyProcessor(BaseParallelProcessor):
            def increment_progressbar(self, queue, /, files = 0, documents = 0):   # we use two progress bars
                return super().increment_progressbar(queue, files=files, documents=documents)
        ```
        """
        return super().increment_progressbar(queue, files=files, documents=documents)

    def _get_all_paths(self) -> Tuple[List[MultiPath], List[MultiPath], List[MultiPath]]:
        """Get all paths to process using prefixes provided"""
        all_src = [MultiPath.parse(f"{self.source_prefix.as_str}/{i}") for i in range(PARTITIONS)]
        all_dst = [MultiPath.parse(f"{self.destination_prefix}/{i}.jsonl.gz") for i in range(PARTITIONS)]
        all_meta = [MultiPath.parse(f"{self.metadata_prefix}/{i}.meta") for i in range(PARTITIONS)]

        return all_src, all_dst, all_meta


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-s", "--source-prefix", type=str, default="tiiuae/falcon-refinedweb")
    ap.add_argument("-p", "--parallel", type=int, default=1)
    opts = ap.parse_args()

    HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN", None)
    multiprocessing.set_start_method("spawn")
    dl = FalconDownloader(
        source_prefix=opts.source_prefix,
        destination_prefix="s3://ai2-llm/pretraining-data/sources/falcon-refinedweb/v0/documents",
        metadata_prefix="s3://ai2-llm/pretraining-data/sources/falcon-refinedweb/v0/metadata",
        num_processes=opts.parallel,
    )
    dl(hf_access_token=HF_ACCESS_TOKEN)
