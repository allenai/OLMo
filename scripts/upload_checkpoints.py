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
import math
import re
from pathlib import Path
from dataclasses import dataclass
from huggingface_hub import HfApi, create_branch, list_repo_refs
from tqdm import tqdm

# Stage-specific model training configurations
# TODO: Read from config.yaml of non-converted checkpoint dir instead of hardcode?
MODEL_CONFIG = {
    "OLMo-2-0425-1B-early": {
        "stage1": {
            "sequence_length": 4096,
            "global_batch_size": 512,
        },
    },
}

@dataclass
class CheckpointInfo:
    path: Path
    step: int
    is_main: bool
    seq_len: int
    batch_size: int

def compute_tokens_b(step: int, seq_len: int, batch_size: int) -> int:
    total_tokens = step * batch_size * seq_len
    return math.ceil(total_tokens / 1_000_000_000)

def get_branch_name(step: int, is_main: bool, stage: str, seq_len: int, batch_size: int) -> str:
    if is_main:
        return "main"
    return f"{stage}-step{step}-tokens{compute_tokens_b(step, seq_len, batch_size)}B"

def upload_checkpoint(info: CheckpointInfo, 
                      repo_id: str, 
                      api: HfApi, 
                      stage: str, 
                      dry_run: bool = False, 
                      overwrite_branch: bool = False):
    
    branch = get_branch_name(info.step, 
                             info.is_main, 
                             stage, 
                             info.seq_len, 
                             info.batch_size)

    if dry_run:
        files = sorted([f.name for f in info.path.iterdir() if f.is_file()])
        return f"{info.step}\t→ {branch}\t{info.path}\n  Files: {', '.join(files)}"

    # Check if the branch already exists (and --overwrite_branch is not set)
    refs = list_repo_refs(repo_id, repo_type="model")
    existing_branches = {ref.name for ref in refs.branches}
    if branch in existing_branches and not overwrite_branch:
        raise RuntimeError(f"Branch '{branch}' already exists in {repo_id}. Use --overwrite_branch to overwrite.")

    if branch != "main":
        create_branch(repo_id=repo_id, branch=branch, repo_type="model", exist_ok=True)

    # TODO: See about using upload_large_folder
    api.upload_folder(
        folder_path=str(info.path),
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"Upload checkpoint step {info.step} to branch '{branch}'",
    )

    print(f"Uploaded: {info.path} → https://huggingface.co/{repo_id}/tree/{branch}")
    return branch

def parse_checkpoints(checkpoints_root: Path, main_step: str, seq_len: int, batch_size: int):
    checkpoints = []
    step_range = [] # to track min/max steps for logging

    def _is_main(step):
        return main_step is not None and main_step.lower() != "none" and int(main_step) == step

    # Case 1: Handle single checkpoint directory named like "stepXXXX-hf" or "stepXXX-unsharded"
    # TODO: Not sure if I want to keep this / use match but was helpful for now
    match = re.match(r"step(\d+)-(unsharded(-hf)?|hf)$", checkpoints_root.name)
    if match and checkpoints_root.is_dir():
        step = int(match.group(1))
        step_range.append(step)
        
        return [CheckpointInfo(
            path=checkpoints_root,
            step=step,
            is_main=_is_main(step),
            seq_len=seq_len,
            batch_size=batch_size,
        )], step, step

    
    # Case 2: Handle directory of multiple checkpoint directories
    for folder in sorted(checkpoints_root.iterdir()):
        if not folder.is_dir():
            continue
        match = re.match(r"step(\d+)-(unsharded(-hf)?|hf)$", folder.name)
        if not match:
            continue
        
        step = int(match.group(1))
        step_range.append(step)
        checkpoints.append(CheckpointInfo(
            path=folder,
            step=step,
            is_main=_is_main(step),
            seq_len=seq_len,
            batch_size=batch_size,
        ))

    if not checkpoints:
        print(f"WARNING: No valid checkpoints found in {checkpoints_root}")

    return checkpoints, min(step_range, default=0), max(step_range, default=0)

def main():
    parser = argparse.ArgumentParser(description="Upload OLMo checkpoints to Hugging Face Hub (1 branch per checkpoint)")

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

    args = parser.parse_args()

    if args.model_name not in MODEL_CONFIG:
        raise ValueError(f"Model '{args.model_name}' is not defined in MODEL_CONFIG")
    if args.stage not in MODEL_CONFIG[args.model_name]:
        raise ValueError(f"Stage '{args.stage}' is not defined for model '{args.model_name}'")

    config = MODEL_CONFIG[args.model_name][args.stage]
    seq_len = config["sequence_length"]
    batch_size = config["global_batch_size"]

    checkpoints_root = Path(args.checkpoints_dir)
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoints_dir}")

    # Enables Hugging Face XET for faster uploads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    api = HfApi()

    dry_run_log = []
    success_log = []
    fail_log = []

    checkpoint_infos, step_min, step_max = parse_checkpoints(checkpoints_root, args.main_step, seq_len, batch_size)

    for info in tqdm(checkpoint_infos, desc="Uploading checkpoints", disable=args.dry_run):
        try:
            result = upload_checkpoint(info, args.repo_id, api, args.stage, args.dry_run, args.overwrite_branch)
            if args.dry_run:
                dry_run_log.append(result)
            else:
                files = sorted([f.name for f in info.path.iterdir() if f.is_file()])
                success_log.append(f"{info.step}\t→ {get_branch_name(info.step, info.is_main, args.stage, seq_len, batch_size)}\n  Files: {', '.join(files)}")
        except Exception as e:
            print(f"Failed to upload step {info.step}: {e}")
            fail_log.append(f"{info.step}\t{info.path}\t{e}")

    suffix = f"{args.stage}_step{step_min}_to_step{step_max}"

    if args.dry_run:
        Path(f"{args.model_name}_{suffix}_dry_run.txt").write_text("\n".join(dry_run_log))
        print(f"\nDry run summary written to: {args.model_name}_{suffix}_dry_run.txt")
    else:
        if success_log:
            Path(f"{args.model_name}_{suffix}_uploaded.txt").write_text("\n".join(success_log))
        if fail_log:
            Path(f"{args.model_name}_{suffix}_failed.txt").write_text("\n".join(fail_log))
        print(f"\nUpload complete. Successes: {len(success_log)}  Failures: {len(fail_log)}")

if __name__ == "__main__":
    main()

