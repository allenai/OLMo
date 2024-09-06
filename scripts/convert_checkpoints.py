# This script requires to be run at the root level.
# Requires the AWS CLI and Beaker Gantry to be installed and configured.


import argparse
import subprocess

# Beaker secret keys
AWS_ACCESS_KEY_ID = 'JENA_AWS_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY = 'JENA_AWS_SECRET_ACCESS_KEY'

SANITY_CHECK = False

def convert_checkpoint(checkpoint_paths):

    for cp in checkpoint_paths:
        retain_path_name = cp.replace('s3://', '').strip('/')
        load_dir = "/data/input"
        weka_loc = f"{load_dir}/{retain_path_name}-hf/"
        log_file = "log.txt"

        cmd = f"gantry run " \
              f"--description 'Converting {cp}' " \
              f"--allow-dirty " \
              f"--no-python " \
              f"--workspace ai2/cheap-decisions " \
              f"--priority normal " \
              f"--gpus 0 " \
              f"--preemptible " \
              f"--cluster 'ai2/jupiter-cirrascale-2' " \
              f"--budget ai2/oe-eval " \
              f"--env-secret AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID} " \
              f"--env-secret AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY} " \
              f"--shared-memory 10GiB " \
              f"--weka=oe-eval-default:{load_dir} " \
              f"--yes " \
              f"-- /bin/bash -c python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir '{cp}' --destination-dir '{weka_loc}' --keep-olmo-artifacts"

        #f"--mount weka://oe-eval-default={load_dir} "
            # FIX THIS
        if SANITY_CHECK:
            print(cmd)
        else:
            try:
                with open(log_file,'w') as fout:
                    subprocess.run(cmd, shell=True, check=True, stdout=fout, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(e.output)


def read_checkpoints(f):
    with open(f,'r') as fin:
        checkpoints = [line for line in f if line and line != '']
    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Unshard checkpoint and convert to HF format. Run via Gantry. Invoke this script from the root of the OLMo repo."
    )

    group_batch = parser.add_mutually_exclusive_group(required=True)
    group_batch.add_argument("--checkpoint_path", help="path to sharded checkpoint", type=str)
    group_batch.add_argument("--checkpoint_path_file", help="file that lists sharded checkpoint paths (batch run option)", type=str)

    args = parser.parse_args()

    if args.checkpoint_path is not None:
        convert_checkpoint([args.checkpoint_path])
    else:
        convert_checkpoint(read_checkpoints(args.checkpoint_path_file))


if __name__ == "__main__":
    main()
