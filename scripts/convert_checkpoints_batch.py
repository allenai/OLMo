"""
Modification of s3_unshard_to_hf.py
Wrapper for hf_olmo/convert_olmo_to_hf.py

Takes a model checkpoint stored on S3, unshards, and converts to HF format.
Saves the converted checkpoints to weka.
Requires AWS CLI to be installed and configured.
"""

import argparse
import subprocess
import os
import json

SANITY_CHECK = True

def convert_checkpoint(cps, load_dir="/data/input"):
    cps = expand_paths(cps)
    processed = []

    for checkpoint_path in cps:
        # Convert to old-style checkpoint.

        retain_path_name = checkpoint_path.replace('s3://', '').strip('/')
        weka_loc = f"{load_dir}/{retain_path_name}-hf/"

        # Check if the output location is already there. If not, do the conversion.
        if os.path.exists(weka_loc):
            conversion = 'existing'
        else:
            conversion = 'new'
            conversion_cmd = f"python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir '{checkpoint_path}' --destination-dir '{weka_loc}' --keep-olmo-artifacts"

            if SANITY_CHECK:
                print(conversion_cmd)
            else:
                subprocess.run(conversion_cmd, shell=True, check=True)

        processed.append({
            'unproccessed_path': checkpoint_path,
            'converted_path': weka_loc,
            'convertion': conversion})

    with open('/data/input/jenah/log.jsonl','a+') as fout:
        for p in processed:
            fout.write(json.dumps(p)+'\n')

def expand_paths(cps):
    expanded = []
    for cp in cps:
        segs = cp.split('*')
        prefix = 's3://ai2-llm/'
        cmd = f"aws s3 ls --recursive {segs[0]}"
        all_dirs = subprocess.run(cmd, shell=True, check=True, capture_output=True, text = True).stdout
        relevant_dirs = ['/'.join(d.split()[-1].split('/')[:-1]) for d in all_dirs.split() if 'model.pt' in d]
        search_segs = [seg for i, seg in enumerate(segs) if i > 0 and seg != ""]

        print(f"search segments: {search_segs}")

        temp_dirs = relevant_dirs
        if len(search_segs) > 0:
            for s in search_segs:
                temp_dirs = [d for d in temp_dirs if s in d]

        exp = set([f"{prefix}{d}" for d in temp_dirs])
        print(exp)

        expanded += exp
    return expanded


def read_checkpoints(f):
    with open(f, 'r') as fin:
        checkpoints = [line for line in fin if line and line != '']
    return checkpoints


def main():
    parser = argparse.ArgumentParser()

    group_batch = parser.add_mutually_exclusive_group(required=True)
    group_batch.add_argument("--checkpoint-path", help="path to sharded checkpoint", type=str)
    group_batch.add_argument("--checkpoint-path-file", help="file that lists sharded checkpoint paths (batch run option)", type=str)
    parser.add_argument("--weka-load-dir", help='mounted location of weka bucket', default='/data/input', type=str)

    args = parser.parse_args()

    if args.checkpoint_path is not None:
        convert_checkpoint([args.checkpoint_path], args.weka_load_dir)
    else:
        convert_checkpoint(read_checkpoints(args.checkpoint_path_file), args.weka_load_dir)


if __name__ == "__main__":
    main()
