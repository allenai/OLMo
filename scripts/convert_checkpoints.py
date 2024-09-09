# This script requires to be run at the root level.
# Requires the AWS CLI and Beaker Gantry to be installed and configured.


import argparse
import subprocess

# Beaker secret keys
AWS_ACCESS_KEY_ID = 'JENA_AWS_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY = 'JENA_AWS_SECRET_ACCESS_KEY'

SANITY_CHECK = True

def convert_checkpoints(args):
    cmd = f"gantry run " \
          f"--allow-dirty " \
          f"--workspace ai2/cheap-decisions " \
          f"--priority normal " \
          f"--gpus 0 " \
          f"--preemptible " \
          f"--cluster 'ai2/jupiter-cirrascale-2' " \
          f"--budget ai2/oe-eval " \
          f"--env-secret AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID} " \
          f"--env-secret AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY} " \
          f"--shared-memory 10GiB " \
          f"--weka=oe-eval-default:{args.weka_load_dir} " \
          f"--yes "

    if args.checkpoint_path is not None:
        cmd += f"-- /bin/bash -c python convert_checkpoints_batch.py --checkpoint-path '{args.checkpoint_path}' --weka-load-dir {args.weka_load_dir}"
    else:
        cmd += f"-- /bin/bash -c python convert_checkpoints_batch.py --checkpoint-path-file '{args.checkpoint_path_file}' --weka-load-dir {args.weka_load_dir}"

    if SANITY_CHECK:
        print(cmd)
    else:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e.output)


def main():
    parser = argparse.ArgumentParser(
        description="Unshard checkpoint and convert to HF format. Run via Gantry. Invoke this script from the root of the OLMo repo."
    )

    group_batch = parser.add_mutually_exclusive_group(required=True)
    group_batch.add_argument("--checkpoint-path", help="path to sharded checkpoint", type=str)
    group_batch.add_argument("--checkpoint-path-file", help="file that lists sharded checkpoint paths (batch run option)", type=str)
    parser.add_argument("--weka-load-dir", help='mounted location of weka bucket', default='/data/input', type=str)

    args = parser.parse_args()
    convert_checkpoints(args)


if __name__ == "__main__":
    main()
