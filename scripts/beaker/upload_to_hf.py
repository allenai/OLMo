import os
import argparse
import boto3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi
from botocore.exceptions import ClientError
from tqdm import tqdm

from olmo.data.named_data_mixes import DATA_SOURCES, EXTRA_DATA_SOURCES
import shutil

assert set(DATA_SOURCES.keys()).intersection(EXTRA_DATA_SOURCES.keys()) == set(), "Named data mixes should not overlap with extra data sources"
DATA_SOURCES.update(EXTRA_DATA_SOURCES)

print_lock = threading.Lock()  # To keep print statements thread-safe

def parse_args():
    parser = argparse.ArgumentParser(description="Sync S3 to local and upload to Hugging Face Hub")
    parser.add_argument("--named-data-mix", required=True, help="Named data mix to download and upload")
    parser.add_argument("--s3-bucket", help="S3 bucket name", default="ai2-llm")
    parser.add_argument("--local-dir", required=True, help="Local folder to store downloaded files")
    parser.add_argument("--hf-repo-id", required=True, help="Hugging Face repo ID (e.g. username/dataset)")
    parser.add_argument("--hf-repo-type", default="dataset", choices=["dataset", "model"], help="Type of Hugging Face repo")
    parser.add_argument("--num-download-workers", type=int, default=8, help="Number of threads to use for S3 download")
    return parser.parse_args()

def should_download(s3_obj, local_dir):
    local_path = os.path.join(local_dir, s3_obj)

    return not os.path.exists(local_path)

def download_file(s3_client, bucket_name, s3_key, local_dir):
    local_path = os.path.join(local_dir, s3_key)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True, s3_key
    except ClientError as e:
        return False, f"{s3_key} ({e})"

def parallel_download_s3(bucket_name, prefixes, local_dir, s3_client, num_workers):
    files_to_download = [f for f in prefixes if should_download(f, local_dir)]

    print(f"Total files: {len(files_to_download)} | To download: {len(files_to_download)}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_file, s3_client, bucket_name, s3_key, local_dir): s3_key
            for s3_key in files_to_download
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading from S3"):
            success, msg = future.result()
            with print_lock:
                if not success:
                    print(f"[ERROR] Failed to download: {msg}")

def upload_to_huggingface(folder_path, repo_id, repo_type):
    print(f"Uploading to Hugging Face hub at {repo_id}...")
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.upload_large_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

def main():
    args = parse_args()

    s3 = boto3.client("s3")

    prefixes = DATA_SOURCES[args.named_data_mix]

    parallel_download_s3(
        bucket_name=args.s3_bucket,
        prefixes=prefixes,
        local_dir=args.local_dir,
        s3_client=s3,
        num_workers=args.num_download_workers
    )

    upload_to_huggingface(
        folder_path=args.local_dir,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type
    )

    # Clean up the local directory after upload
    if os.path.exists(args.local_dir):
        os.system(f"rm -rf {args.local_dir}")
    print(f"Deleted local directory: {args.local_dir}")

if __name__ == "__main__":
    main()
