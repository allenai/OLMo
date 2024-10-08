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
import time
import boto3
import json
from pathlib import Path

from gantry import RESULTS_DIR

# possible converted locations.
# "self" is the target location where the converted model would be saved
# key: template, value: description
# template: MUST obey .format(load_dir, retain_path_name)

WEKA_CHECK_LOCATIONS_PREFIXES = {
    "{}/{}-hf/pytorch_model.bin": 'self',
    "{}/ianm/{}-hf/pytorch_model.bin": "ian's"
}

def convert_checkpoint(cps, load_dir="/data/input", sanity_check=False, weka_prefix="/weka", save_to_weka=False):
    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    cps = expand_paths(cps, s3_client)

    print(f">>> Total of {len(cps)} paths to process. <<<", flush=True)

    processed = {}

    # Convert to old-style checkpoint.
    for checkpoint_path in cps:
        print(f"\n\n------------------------------------------------------------", flush=True)
        print(f"\nProcessing Checkpoint: {checkpoint_path}\n", flush=True)

        error = ""
        converted_path = ""
        existing_location = ""
        conversion_status = ""

        # sort out paths, bucket names, and so on ...
        path_bits = checkpoint_path.strip('/').replace('s3://', '').split('/')
        s3_bucket_name = path_bits[0]
        s3_prefix = '/'.join(path_bits[1:])
        temp_path = '/'.join(path_bits) #checkpoint_path.replace('s3://', '').strip('/')
        local_path = f"{load_dir}/{temp_path}-hf/"

        # the converted model may already exist in local_path or in
        path_found = False
        potential_existing_locations = [l.format(load_dir,temp_path) for l in WEKA_CHECK_LOCATIONS_PREFIXES]
        for loc in potential_existing_locations:
            if os.path.exists(loc):
                existing_location = loc.replace('/pytorch_model.bin','')
                path_found = True
                break

        # if one of the potential existing location has converted model in it then use that
        if path_found:
            # then there is no conversion to do.
            conversion_status = 'existing'
            converted_path = existing_location
            print(f"Converted Checkpoint Found: {converted_path}\n", flush=True)
        else:
            s3_bucket = s3_resource.Bucket(s3_bucket_name)
            s3_hf_exists = s3_path_exists(s3_bucket, s3_prefix, s3_bucket_name)

            # if s3 already has a location for converted model then use that
            if s3_hf_exists is not None:
                path_found = True
                print(f"Converted Checkpoint Found: {s3_hf_exists}", flush=True)

                # if save to weka flag is passed, then download the s3 converted model to the local path
                if save_to_weka:
                    copy_s3_to_local(s3_bucket, s3_prefix, local_path, local_path.replace(load_dir,weka_prefix), sanity_check)
                    conversion_status = 'existing-downloaded'
                    converted_path = local_path
                else:
                    conversion_status = 'existing'
                    converted_path = s3_hf_exists

        # if no existing conversions are found then process and save to local path
        if not path_found:
            conversion_status = 'new'
            converted_path = local_path
            conversion_cmd = f"python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir '{checkpoint_path}' --destination-dir '{local_path}' --tokenizer 'allenai/gpt-neox-olmo-dolma-v1_5'  --cleanup-local-dir"

            if sanity_check:
                print('SANITY CHECK MODE (not running the conversion)')
                print(conversion_cmd + '\n')
            else:
                try:
                    subprocess.run(conversion_cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    error = e.output ### NOT ACTUALLY WORKING CORRECTLY. FIX THIS (not catching config not found error)
                    conversion_status = 'error'
                    converted_path = ""

        # Keep info for log.jsonl
        local_log = {
            'unprocessed_path': checkpoint_path,
            'converted_path': converted_path.replace(load_dir,weka_prefix),
            'conversion': conversion_status,
            'date_time': time.strftime('%b-%d-%Y_%H%M', time.localtime()),
            'error': error
        }

        # output model checkpoint location for eval scripts
        curr = Path(converted_path)
        parent = curr.parent
        if parent.name not in processed:
            processed[parent.name] = {
                'model_name': parent.name,
                'checkpoints_location': str(parent).replace(load_dir,weka_prefix),
                'revisions': [curr.name]
            }
        else:
            processed[parent.name]['revisions'].append(curr.name)

        # Output Log
        if not sanity_check:
            with open(os.path.join(RESULTS_DIR, 'log.jsonl'), 'a+') as fout:
                fout.write(json.dumps(local_log) + '\n')

    # Output checkpoint location for eval scripts
    if not sanity_check:
        with open(os.path.join(RESULTS_DIR, 'model_checkpoints.jsonl'), 'w') as fout:
            for _, p in processed.items():
                fout.write(json.dumps(p) + '\n')


def s3_path_exists(bucket, prefix, bucket_name):
    # look for pytorch_model.bin in directories ending with -hf or -hf-olmo.
    objs = list(bucket.objects.filter(Prefix=prefix + '-hf/pytorch_model.bin'))
    if len(objs) > 0:
        return f"s3://{bucket_name}/{prefix}-hf"
    else:
        objs2 = list(bucket.objects.filter(Prefix=prefix + '-hf-olmo/pytorch_model.bin'))
        return f"s3://{bucket_name}/{prefix}-hf-olmo" if (len(objs2) > 0) else None


def copy_s3_to_local(bucket, prefix, local_path, display_name, sanity_check):
    # if not os.path.exists(os.path.dirname(local_path)):
    print(f"Downloading checkpoint to {display_name}\n", flush=True)
    if not sanity_check:
        try:
            os.makedirs(local_path)
        except:
            pass
        print(prefix)
        print(local_path)
        bucket.download_file(bucket, prefix, local_path)  # save to same path


def expand_paths(cps, s3):
    expanded = []

    for cp in cps:
        bucket = cp.split('/')[2]
        segs = cp.split('*')
        prefix = segs[0].replace('s3://'+bucket+'/', '')

        relevant_dirs = []
        skip_parent = []

        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        contents = {obj["Key"]:str(Path(obj['Key']).parent) for page in page_iterator for obj in page['Contents']}
        paths = set(contents.values())

        for path in contents:
            p = Path(path)
            parent = str(p.parent)
            grandpa = str(p.parent.parent)

            if parent in relevant_dirs or parent in skip_parent:
                continue
            if p.parent.name in ['optim', 'train','model']:
                if f"{grandpa}-unsharded" in paths:
                    # skip condition
                    skip_parent.append(parent)
                    continue
                else:
                    relevant_dirs.append(grandpa)
            elif p.name == 'model.pt':
                relevant_dirs.append(parent)

        search_segs = [seg for i, seg in enumerate(segs) if i > 0 and seg != ""]

        # subselect the directory with remaining segments (for multiple wildcard *)
        temp_dirs = relevant_dirs
        if len(search_segs) > 0:
            for s in search_segs:
                temp_dirs = [d for d in temp_dirs if s in d]

        exp = set([f"s3://{bucket}/{d}" for d in temp_dirs])

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
    parser.add_argument("--weka-prefix", help='weka directory prefix for output', default='/weka', type=str)
    parser.add_argument("--sanity-check", help='print what would be run; do not actually run conversion', action='store_true')
    parser.add_argument("--save-to-weka", help='if checkpoints are found on s3, save them to loaded weka dir', action='store_true')

    args = parser.parse_args()

    if args.checkpoint_path is not None:
        convert_checkpoint([args.checkpoint_path], load_dir=args.weka_load_dir, sanity_check=args.sanity_check, weka_prefix=args.weka_prefix, save_to_weka=args.save_to_weka)
    else:
        convert_checkpoint(read_checkpoints(args.checkpoint_path_file), load_dir=args.weka_load_dir, sanity_check=args.sanity_check, weka_prefix=args.weka_prefix, save_to_weka=args.save_to_weka)


if __name__ == "__main__":
    main()
