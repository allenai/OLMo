# a script that takes a path that has a bunch of unsharded checkpoints and finds all the unsharded checkpoints that have not been converted to hf checkpoints yet

import argparse
import os

def main(args):
    # get all subdirectories named step*-unsharded

    non_converted = set(subdir for subdir in os.listdir(args.path) if subdir.endswith("-unsharded"))

    # get all subdirectories named step*-unsharded-hf
    converted = set(subdir for subdir in os.listdir(args.path) if subdir.endswith("-unsharded-hf"))

    # find the difference between the two lists
    need_conversion = [subdir for subdir in non_converted if subdir.replace("-unsharded", "-unsharded-hf") not in converted]

    print(" ".join(need_conversion))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="find all the unsharded checkpoints that have not been converted to hf checkpoints yet")
    parser.add_argument("--path", type=str, help="Path to the directory that contains the checkpoints to convert")
    args = parser.parse_args()
    main(args)