"""
Convenience script to take a model checkpoint stored on S3, unshard, and convert to HF
format. Requires the AWS CLI to be installed and configured.
"""

import argparse
import pathlib
import subprocess
import shutil


def make_parser():
    parser = argparse.ArgumentParser(
        description="Unshard S3 checkpoint and convert to HF format. Invoke this script from the root of the OLMo repo."
    )
    parser.add_argument(
        "--sharded_bucket", help="S3 bucket with sharded checkpoint.", type=str
    )
    parser.add_argument(
        "--unsharded_bucket",
        help="S3 bucket to save the unsharded checkpoint.",
        type=str,
    )
    parser.add_argument(
        "--hf_bucket", help="S3 bucket to save the HF-converted checkpoint.", type=str
    )
    parser.add_argument(
        "--tmp_dir",
        help="""Temporary directory to store checkpoints locally. This will be deleted
        if everything runs successfully, but will keep files around otherwise to avoid
        re-downloads when possible.""",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If given, don't show progress for AWS commands.",
    )
    return parser


def aws_copy(src, dest, quiet):
    base = "aws s3 cp --recursive"
    if quiet:
        base += " --quiet"
    cmd = f"{base} {src} {dest}"

    return cmd


def s3_unshard_to_hf(args):
    # Set directories
    sharded_dir = args.tmp_dir / "sharded"
    unsharded_dir = args.tmp_dir / "unsharded"
    hf_dir = args.tmp_dir / "hf"
    hf_dir.mkdir()

    # Download sharded checkpoint.
    print("Downloading sharded checkpoint from S3.")
    download_cmd = aws_copy(args.sharded_bucket, sharded_dir, args.quiet)
    subprocess.run(download_cmd, shell=True, check=True)

    # Unshard.
    print("Unsharding.")
    unshard_cmd = (
        f"python scripts/unshard.py {sharded_dir} {unsharded_dir} --model-only"
    )
    subprocess.run(unshard_cmd, shell=True, check=True)

    # Convert to HF
    print("Converting to HF.")
    hf_cmd = f"python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir {unsharded_dir}"
    subprocess.run(hf_cmd, shell=True, check=True)

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
    print("Uploading unsharded and HF files back to S3.")
    upload_unsharded_cmd = aws_copy(unsharded_dir, args.unsharded_bucket, args.quiet)
    subprocess.run(upload_unsharded_cmd, shell=True, check=True)

    upload_hf_cmd = aws_copy(hf_dir, args.hf_bucket, args.quiet)
    subprocess.run(upload_hf_cmd, shell=True, check=True)


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.tmp_dir.exists():
        raise ValueError(
            f"Temporary directory {args.tmp_dir} already exists; refusing to write."
        )
    args.tmp_dir.mkdir()

    s3_unshard_to_hf(args)

    # Clear out temp dir if we got here (everything ran without error).
    shutil.rmtree(args.tmp_dir)


if __name__ == "__main__":
    main()
