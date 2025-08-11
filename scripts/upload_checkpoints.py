"""
Flexible script to upload OLMo checkpoints to Hugging Face Hub.
Supports uploading either a single checkpoint or a directory of them.
Uses Hugging Face XET for efficient uploads: https://huggingface.co/docs/huggingface_hub/guides/upload#faster-uploads
To use XET, do: pip install -U "huggingface_hub[hf_xet]"

Example Usage:
  python upload_checkpoints.py \
    --checkpoints_dir /path/to/all-checkpoints \
    --model_name OLMo-2-0425-1B-early \
    --repo_id allenai/OLMo-2-0425-1B-early \
    --stage stage1 \
    --main_step 30000

Add --dry_run to preview uploads and write logs instead of uploading.
"""

import argparse
import os
import sys
import math
import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

from huggingface_hub import HfApi, create_branch, list_repo_refs
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "OLMo-2-0425-1B-early": {
        "stage1": {
            "sequence_length": 4096,
            "global_batch_size": 512,
        },
    },
}

RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

@dataclass
class CheckpointInfo:
    path: Path
    step: int
    is_main: bool
    seq_len: int
    batch_size: int

 ### ----- Retrieve Checkpoint Info -----
def compute_tokens_b(step: int, seq_len: int, batch_size: int) -> int:
    total_tokens = step * batch_size * seq_len
    return math.ceil(total_tokens / 1_000_000_000)

def get_branch_name(step: int, is_main: bool, stage: str, seq_len: int, batch_size: int) -> str:
    if is_main:
        return "main"
    return f"{stage}-step{step}-tokens{compute_tokens_b(step, seq_len, batch_size)}B"

### ----- Restart and Retry Utilites -----
def get_remote_file_sizes(api: HfApi, repo_id: str, branch: str) -> Dict[str, int]:
    """Return a mapping of {path_in_repo: size_in_bytes} for files on a branch.
    If the branch doesn't exist, return an empty dict.
    """
    try:
        tree = api.list_repo_tree(repo_id=repo_id, repo_type="model", revision=branch, recursive=True)
    
    # Handle branch not found - doesn't exist yet
    except HfHubHTTPError as e:
        if getattr(e.response, "status_code", None) == 404:
            return {}
        raise

    sizes: Dict[str, int] = {}
    for item in tree:
        path = getattr(item, "path", None) or getattr(item, "rfilename", None)
        size = getattr(item, "size", None)
        if path is not None and size is not None:
            sizes[path] = int(size)
    return sizes

def local_files_with_sizes(root: Path) -> List[Tuple[Path, int]]:
    """Return a list of (file_path, size_in_bytes) under the given root directory."""
    files_and_sizes: List[Tuple[Path, int]] = []

    for file_path in root.rglob("*"): 
        if file_path.is_file():
            try:
                files_and_sizes.append((file_path, file_path.stat().st_size))
            except FileNotFoundError:
                logger.error(f"Local file disappeared while scanning: {file_path}")
                raise
    return files_and_sizes

def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (RequestsConnectionError, TimeoutError)):
        return True
    if isinstance(exc, HfHubHTTPError):
        try:
            status = exc.response.status_code  # type: ignore[attr-defined]
            return status in RETRYABLE_HTTP_STATUSES
        except Exception:
            return False
    return False

### ----- Upload Checkpoint Directories or Files -----
def upload_with_retries(api: HfApi, upload_func, upload_args: dict, operation_name: str, max_retries: int, retry_sleep: float) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            upload_func(**upload_args)
            return
        except Exception as e:
            if is_retryable_error(e) and attempt < max_retries:
                logger.warning(f"Retryable error during {operation_name} (attempt {attempt}/{max_retries}): {e}. Retrying in {retry_sleep}s...")
                time.sleep(retry_sleep)
            else:
                logger.error(f"Failed {operation_name}: {e}")
                raise

def full_dir_upload(branch: str, existing_branches: set, overwrite_branch: bool, remote_sizes: dict) -> bool:
    """Determine if we should upload the entire folder or individual file(s)."""
    return (
        branch not in existing_branches or 
        overwrite_branch or 
        not remote_sizes  # empty remote directory
    )

def upload_checkpoint(info: CheckpointInfo, repo_id: str, api: HfApi, stage: str, 
                      dry_run: bool, overwrite_branch: bool, max_retries: int, retry_sleep: float) -> str:
    
    branch = get_branch_name(info.step, info.is_main, stage, info.seq_len, info.batch_size)
    
    if dry_run:
        files = sorted([f.name for f in info.path.iterdir() if f.is_file()])
        return f"{info.step}\t→ {branch}\t{info.path}\n  Files: {', '.join(files)}"

    # Get repository state
    refs = list_repo_refs(repo_id, repo_type="model")
    existing_branches = {ref.name for ref in refs.branches}
    remote_sizes = {} if branch not in existing_branches else get_remote_file_sizes(api, repo_id, branch)
    
    # Create branch if needed
    if branch not in existing_branches and branch not in ["main"]:
        create_branch(repo_id=repo_id, branch=branch, repo_type="model", exist_ok=True)
    
    # Decide upload strategy (file or full directory)
    if full_dir_upload(branch, existing_branches, overwrite_branch, remote_sizes):
        upload_with_retries(
            api=api,
            upload_func=api.upload_folder,
            upload_args={
                "folder_path": str(info.path),
                "repo_id": repo_id,
                "repo_type": "model", 
                "revision": branch,
                "commit_message": f"Upload checkpoint step {info.step} to branch '{branch}'"
            },
            operation_name=f"folder upload to {branch}",
            max_retries=max_retries,
            retry_sleep=retry_sleep
        )
        logger.info(f"UPLOADED FOLDER: {info.path} → https://huggingface.co/{repo_id}/tree/{branch}")
    else:
        # Upload individual files
        _partial_dir_upload(info, repo_id, api, branch, remote_sizes, overwrite_branch, max_retries, retry_sleep)
    
    return branch

def _partial_dir_upload(info: CheckpointInfo, repo_id: str, api: HfApi, branch: str, 
                               remote_sizes: dict, overwrite_branch: bool, max_retries: int, retry_sleep: float) -> None:
    """Handle individual file uploads with size checking."""
    for file_path, size in tqdm(local_files_with_sizes(info.path), desc=f"step {info.step}", leave=False):
        rel_path = file_path.relative_to(info.path).as_posix()
        
        # Skip if file exists with same size
        if rel_path in remote_sizes:
            if remote_sizes[rel_path] == size:
                logger.info(f"SKIP (exists, same size): {branch}:{rel_path}")
                continue
            if not overwrite_branch:
                raise RuntimeError(
                    f"File exists with different size: {branch}:{rel_path} "
                    f"(remote={remote_sizes[rel_path]}, local={size}). "
                    "Use --overwrite_branch to overwrite."
                )
        
        # Upload the file
        upload_with_retries(
            api=api,
            upload_func=api.upload_file,
            upload_args={
                "path_or_fileobj": str(file_path),
                "path_in_repo": rel_path,
                "repo_id": repo_id,
                "repo_type": "model",
                "revision": branch,
                "commit_message": f"Upload {rel_path} for checkpoint step {info.step} into '{branch}'"
            },
            operation_name=f"file upload {branch}:{rel_path}",
            max_retries=max_retries,
            retry_sleep=retry_sleep
        )
        logger.info(f"UPLOADED: {branch}:{rel_path} ({size} bytes)")
    
    logger.info(f"Uploaded checkpoint {info.path} → https://huggingface.co/{repo_id}/tree/{branch}")

# -----  -----
def parse_checkpoints(checkpoints_root: Path, main_step: str, seq_len: int, batch_size: int):
    checkpoints: List[CheckpointInfo] = []
    step_range: List[int] = []

    def _is_main(step: int) -> bool:
        return main_step is not None and main_step.lower() != "none" and int(main_step) == step

    # just the hf converted checkpoints
    pattern = r"step([0-9]+)-hf$"

    mroot = re.match(pattern, checkpoints_root.name)
    if mroot and checkpoints_root.is_dir():
        step = int(mroot.group(1))
        step_range.append(step)
        return [
            CheckpointInfo(
                path=checkpoints_root,
                step=step,
                is_main=_is_main(step),
                seq_len=seq_len,
                batch_size=batch_size,
            )
        ], step, step

    for folder in sorted(checkpoints_root.iterdir()):
        if not folder.is_dir(): # Skip non-directories
            continue
        m = re.match(pattern, folder.name)
        if not m:
            continue
        step = int(m.group(1))
        step_range.append(step)
        checkpoints.append(
            CheckpointInfo(
                path=folder,
                step=step,
                is_main=_is_main(step),
                seq_len=seq_len,
                batch_size=batch_size,
            )
        )

    if not checkpoints:
        logger.warning(f"No valid checkpoints found in {checkpoints_root}")

    return checkpoints, (min(step_range) if step_range else 0), (max(step_range) if step_range else 0)

def main():
    parser = argparse.ArgumentParser(description="Upload OLMo checkpoints (upload_folder fast-path + per-file restart)")

    parser.add_argument(
        "--repo_id",            
        type=str,   
        default="allenai/OLMo-2-0425-1B-early",   
        help="Hugging Face model repo"
    )
    parser.add_argument(
        "--checkpoints_dir",    
        type=str,   
        required=True,                           
        help="Path to checkpoint dir or directory of them"
    )
    parser.add_argument(
        "--model_name",         
        type=str,   
        required=True,                           
        help="Key from MODEL_CONFIG for model + stage params"
    )
    parser.add_argument(
        "--stage",              
        type=str,   
        required=True,  
        choices=["stage1", "stage2"], 
        help="Training stage (e.g., stage1)"
    )
    parser.add_argument(
        "--main_step",          
        type=str,   
        default=None,                             
        help="Step to upload as main branch, or 'none' to skip"
    )
    parser.add_argument(
        "--overwrite_branch",   
        action="store_true",                                 
        help="If set, will overwrite existing branch if it exists"
    )
    parser.add_argument(
        "--dry_run",            
        action="store_true",                                 
        help="Preview uploads without actually uploading. Says which files would be uploaded and their branch names. Writes to log file."
    )

    # Retry controls (simple)
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries for retryable errors")
    parser.add_argument("--retry_sleep", type=float, default=5.0, help="Seconds to sleep between retries")

    # Verbosity
    parser.add_argument("--verbose", action="store_true", help="More verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.model_name not in MODEL_CONFIG:
        logger.error(f"Model '{args.model_name}' is not defined in MODEL_CONFIG")
        raise SystemExit(2)
    if args.stage not in MODEL_CONFIG[args.model_name]:
        logger.error(f"Stage '{args.stage}' is not defined for model '{args.model_name}'")
        raise SystemExit(2)

    config = MODEL_CONFIG[args.model_name][args.stage]
    seq_len = config["sequence_length"]
    batch_size = config["global_batch_size"]

    checkpoints_root = Path(args.checkpoints_dir)
    if not checkpoints_root.exists():
        logger.error(f"Checkpoint path not found: {args.checkpoints_dir}")
        raise SystemExit(2)

    # Enable faster uploads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    api = HfApi()

    checkpoint_infos, step_min, step_max = parse_checkpoints(checkpoints_root, args.main_step, seq_len, batch_size)

    for info in tqdm(checkpoint_infos, desc="Uploading checkpoints", disable=args.dry_run):
        branch = upload_checkpoint(
            info,
            args.repo_id,
            api,
            args.stage,
            args.dry_run,
            args.overwrite_branch,
            args.max_retries,
            args.retry_sleep,
        )
        if args.dry_run:
            logger.info(f"DRY RUN: would process {info.path} → branch {branch}")

    if args.dry_run:
        logger.info("Dry run complete")
    else:
        logger.info("All requested checkpoints processed successfully")


if __name__ == "__main__":
    main()