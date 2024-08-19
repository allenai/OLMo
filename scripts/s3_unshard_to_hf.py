"""
Convenience script to take a model checkpoint stored on S3, unshard, and convert to HF
format. Requires the AWS CLI to be installed and configured.

Example usage for `olmo_core`-style checkpoint (circa April 2024):
python scripts/s3_unshard_to_hf.py \
    --sharded_bucket s3://ai2-llm/checkpoints/OLMo-medium/mitchish7/step239000 \
    --unsharded_bucket s3://ai2-llm/checkpoints/OLMo-medium/mitchish7/step239000-unsharded \
    --hf_bucket s3://ai2-llm/checkpoints/OLMo-medium/mitchish7/step239000-huggingface \
    --type olmo_core \
    --local_dir /net/nfs.cirrascale/allennlp/davidw/tmp/unshard

NOTE: For this to work, you need to install the `OLMo-core` repo as follows:
- Clone https://github.com/allenai/OLMo-core
- Run `pip install -e .[all]`
"""

import argparse
import pathlib
import shutil
import subprocess


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


def aws_copy(src, dest, args):
    base = "aws s3 sync --exclude tmp/*"
    if args.quiet:
        base += " --quiet"
    if args.type == "olmo_core" and args.model_only:
        # Don't copy optimizer and trainer state if we're only unsharding the model.
        base += " --exclude optim/* --exclude train/*"
    cmd = f"{base} {src} {dest}"

    return cmd


def s3_unshard_to_hf(args):
    # Set directories
    sharded_dir = args.local_dir / "sharded"
    unsharded_dir = args.local_dir / "unsharded"
    if args.checkpoint_style == "hf_olmo":
        hf_dir = args.local_dir / "hf-olmo"
    elif args.checkpoint_style == "transformers":
        hf_dir = args.local_dir / "transformers"
    else:
        raise ValueError(f"Unknown checkpoint style: {args.checkpoint_style}.")
    hf_dir.mkdir(exist_ok=True)

    # Either download the unsharded checkpoint, or download sharded and unshard.
    if args.already_unsharded:
        download_cmd = aws_copy(args.unsharded_bucket, unsharded_dir, args)
        subprocess.run(download_cmd, shell=True, check=True)
    else:
        if not args.already_downloaded:
            # Download sharded checkpoint.
            print("Downloading sharded checkpoint from S3.")
            download_cmd = aws_copy(args.sharded_bucket, sharded_dir, args)
            subprocess.run(download_cmd, shell=True, check=True)

        # Unshard.
        print("Unsharding.")
        unshard_cmd = f"python scripts/unshard.py {sharded_dir} {unsharded_dir}"
        # Add a `--type` argument if given.
        if args.type is not None:
            unshard_cmd += f" --type {args.type}"
        if args.model_only:
            unshard_cmd += " --model-only"

        subprocess.run(unshard_cmd, shell=True, check=True)

    # Convert to HF.
    print("Converting to HF.")
    if args.checkpoint_style == "hf_olmo":
        # Convert to old-style checkpoint.
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
    else:
        # Convert to new-style checkpoint.
        hf_cmd = f"""python scripts/convert_olmo_to_hf_new.py \
            --input_dir {unsharded_dir} \
            --output_dir {hf_dir} \
            --tokenizer_json_path olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
            --safe_serialization True \
            --no_tmp_cleanup"""
        subprocess.run(hf_cmd, shell=True, check=True)

    # Upload the unsharded and HF files back to S3.
    print("Uploading files back to S3.")
    if not args.already_unsharded:
        upload_unsharded_cmd = aws_copy(unsharded_dir, args.unsharded_bucket, args)
        subprocess.run(upload_unsharded_cmd, shell=True, check=True)

    upload_hf_cmd = aws_copy(hf_dir, args.hf_bucket, args)
    subprocess.run(upload_hf_cmd, shell=True, check=True)


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
