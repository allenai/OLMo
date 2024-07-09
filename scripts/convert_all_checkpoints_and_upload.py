# a script that takes a s3 path that has a bunch of unsharded checkpoints and then converts them one at a time and uploads them again

import argparse
import os
import boto3

def get_paths_to_convert(args):
    bucket = args.s3_path.split("/")[2]
    path= "/".join(args.s3_path.split("/")[3:])
    s3 = boto3.client('s3')

    # list just the subdirectories of the provided path
    response = s3.list_objects_v2(Bucket=bucket, Prefix=path, Delimiter="/")
    paths = []
    for obj in response['CommonPrefixes']:
        paths.append(obj['Prefix'].split("/")[-2])

    # get the list of unsharded checkpoints, eg step38000-unsharded
    unsharded_paths = [path for path in paths if path.endswith("-unsharded")]
    converted_paths = [path for path in paths if path.endswith("-hf")]
    paths_to_convert = [path for path in unsharded_paths if path+'-hf' not in converted_paths]

    return paths_to_convert

def main(args):
    # get all the paths to convert
    paths = get_paths_to_convert(args)
    print(paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all checkpoints in a s3 path and upload them again")
    parser.add_argument("--s3_path", type=str, help="The s3 path to the checkpoints")
    args = parser.parse_args()
    main(args)