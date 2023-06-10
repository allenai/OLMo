import concurrent.futures
import json
import multiprocessing
import random
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

try:
    import boto3
    import tqdm
    from smashed.utils import (
        MultiPath,
        open_file_for_read,
        open_file_for_write,
        recursively_list_files,
    )
except ImportError:
    print("Missing dependencies. Please run `pip install 'smashed[remote]' tqdm`")


CONFIG_PATH = Path(__file__).parent.parent / "config/v1-small/c4-cleaned/gopher-filtered/dedupe-paragraphs.json"
DOCUMENTS_PREFIX = "pretraining-data/sources/common-crawl/v1-c4-cleaned/documents"
BUCKET_NAME = "ai2-llm"
WORKDIR_PREFIX = Path("/tmp/v1-c4-cleaned")


def parse_options() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument(
        "-c",
        "--config-path",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the config file for the deduper; we will replace paths for the documents",
    )
    ap.add_argument(
        "-d", "--documents-path", type=str, default=DOCUMENTS_PREFIX, help="Path to the documents in the S3 bucket"
    )
    ap.add_argument("-b", "--bucket-name", type=str, default=BUCKET_NAME, help="Name of the S3 bucket")
    ap.add_argument(
        "-p",
        "--max-partition-size",
        type=int,
        default=1024 * 1024 * 1024 * 1024,  # 1 TB
        help="Maximum size of a partition in bytes",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        required=True,
        help="Path to the output directory for the new configs",
    )
    ap.add_argument("-s", "--seed", type=int, default=5051, help="Random seed for the partitioning.")
    ap.add_argument(
        "-w",
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of workers to use for getting file sizes",
    )
    ap.add_argument(
        "-l", "--local-workdir-prefix", type=Path, default=WORKDIR_PREFIX, help="Prefix for the local workdir"
    )
    return ap.parse_args()


def get_s3_object_size(path: str) -> Tuple[str, int]:
    parsed_path = MultiPath.parse(path)

    s3 = boto3.client("s3")
    response = s3.head_object(Bucket=parsed_path.bucket, Key=parsed_path.key)
    size = response["ContentLength"]
    return parsed_path.key, size


def parallel_get_size(paths: List[str], max_workers: int = 1) -> List[Tuple[str, int]]:
    results = []
    with tqdm.tqdm(paths, desc="Getting file sizes", unit=" files") as p:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_s3_object_size, path) for path in paths]
            for future in concurrent.futures.as_completed(futures):
                p.update(1)
                results.append(future.result())
    return results


def customize_config(config: dict, output_cnt: int, documents: List[str], local_workdir_prefix: Path) -> dict:
    config = deepcopy(config)
    config["documents"] = sorted(documents)
    config["bloom_filter"]["file"] = str(local_workdir_prefix / f"{output_cnt}.bloom")
    config["work_dir"]["input"] = str(local_workdir_prefix / f"{output_cnt}.input")
    config["work_dir"]["output"] = str(local_workdir_prefix / f"{output_cnt}.output")
    return config


def main():
    opts = parse_options()

    random.seed(opts.seed)
    documents_path = f"s3://{opts.bucket_name.strip('/')}/{opts.documents_path.lstrip('/')}"
    all_files = list(recursively_list_files(documents_path))
    random.shuffle(all_files)

    with open_file_for_read(opts.config_path, "rt") as f:
        base_config: dict = json.load(f)

    sizes_and_keys = parallel_get_size(all_files, max_workers=opts.workers)

    current_size = 0
    output_cnt = 0
    current_files: List[str] = []

    for key, size in sizes_and_keys:
        if current_size + size > opts.max_partition_size:
            # create a new config, figure out where to write it
            output_path = opts.output_dir / f"{output_cnt}.json"
            current_config = customize_config(
                config=base_config,
                output_cnt=output_cnt,
                documents=current_files,
                local_workdir_prefix=opts.local_workdir_prefix,
            )

            # actually write the config
            with open_file_for_write(output_path, "wt") as f:
                json.dump(current_config, f, indent=4)

            # print stats
            print(f"{output_path}: {current_size:.2e} bytes, {len(current_config['documents'])} files.")

            # reset
            current_size = 0
            current_files = []
            output_cnt += 1

        current_files.append(key)
        current_size += size

    if len(current_files) > 0:
        # in case we have some files left over, write them out
        output_path = opts.output_dir / f"{output_cnt}.json"
        current_config = customize_config(
            config=base_config,
            output_cnt=output_cnt,
            documents=current_files,
            local_workdir_prefix=opts.local_workdir_prefix,
        )

        with open_file_for_write(output_path, "wt") as f:
            json.dump(current_config, f, indent=4)

        print(f"{output_path}: {current_size:.2e} bytes, {len(current_config['documents'])} files.")


if __name__ == "__main__":
    main()
