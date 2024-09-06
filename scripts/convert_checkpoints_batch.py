"""
Modification of s3_unshard_to_hf.py
Wrapper for hf_olmo/convert_olmo_to_hf.py

Takes a model checkpoint stored on S3, unshards, and converts to HF format.
Saves the converted checkpoints to weka.
Requires AWS CLI to be installed and configured.
"""

import argparse
import pathlib
import shutil
import subprocess
import os


def convert_to_hf(args):
    # Ensure local directory exists
    if not os.path.exists(local_file_dir):
        os.makedirs(local_file_dir)

    # Convert to old-style checkpoint.
    hf_cmd = f"python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir {unsharded_dir} --destination-dir {local_file_dir}"
    subprocess.run(hf_cmd, shell=True, check=True)

    # Move to Weka
    if not os.path.exists(weka_file_dir):
        os.makedirs(weka_file_dir)



    # Move the HF files from the unsharded dir to their own.
    for fname in [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        (unsharded_dir / fname).rename(hf_dir / fname)

    # Upload the unsharded and HF files back to S3.
    print("Uploading files back to S3.")
    if not args.already_unsharded:
        upload_unsharded_cmd = aws_copy(unsharded_dir, args.unsharded_bucket, args)
        subprocess.run(upload_unsharded_cmd, shell=True, check=True)

    upload_hf_cmd = aws_copy(hf_dir, args.hf_bucket, args)
    subprocess.run(upload_hf_cmd, shell=True, check=True)

def make_parser():
    parser = argparse.ArgumentParser(
        description="Unshard S3 checkpoint and convert to HF format. Invoke this script from the root of the OLMo repo."
    )
    parser.add_argument("--sharded_bucket", help="S3 bucket with sharded checkpoint.", type=str)
    parser.add_argument(
        "--unsharded_bucket",
        help="S3 bucket to save the unsharded checkpoint.",
        type=str,
    )
    parser.add_argument(
        "--already_downloaded",
        action="store_true",
        help="Use this flag if the unsharded S3 checkpoint is already downloaded, but still needs to be unsharded.",
    )
    parser.add_argument(
        "--already_unsharded",
        action="store_true",
        help="If given, the checkpoint has already been unsharded; just convert to HF.",
    )
    parser.add_argument("--hf_bucket", help="S3 bucket to save the HF-converted checkpoint.", type=str)
    parser.add_argument(
        "--local_dir",
        help="""Directory to store checkpoints locally.""",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--cleanup_local_dir",
        action="store_true",
        help="If given, remove the local directory if everything runs successfully to free up space on NFS.",
    )
    parser.add_argument(
        "--checkpoint_style",
        default="hf_olmo",
        choices=["hf_olmo", "transformers"],
        help="""Checkpoint style. The `transformers` style works with HF transformers as-is, while
             `hf_olmo` relies on the `hf_olmo` package for conversion. In general, use
             `transformers` for external releases and `hf_olmo` for internal model
             development.""",
    )
    parser.add_argument(
        "--hf_olmo",
        action="store_true",
        help="If given, convert to 'hf-olmo' style checkpoints.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If given, don't show progress for AWS commands.",
    )
    parser.add_argument("--type", default=None, help="If given, pass this argument on to `unshard.py`.")
    parser.add_argument("--model_only", action="store_true", help="If given, only unshard the model.")
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    args.local_dir.mkdir(exist_ok=True, parents=True)

    s3_unshard_to_hf(args)

    if args.cleanup_local_dir:
        # Clear out temp dir if we got here (everything ran without error).
        shutil.rmtree(args.tmp_dir)


if __name__ == "__main__":
    main()
