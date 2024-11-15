"""
A Script that creates a bunch of ladder runs over the named-data-mixes and scales provided in files.
e.g. output  will be a bash script like the following:
scripts/beaker/ladder-launch.sh 1 normal --model 1B --data DCLM-baseline --length 5xC --name DCLM-baseline  
scripts/beaker/ladder-launch.sh 1 normal --model 750M --data DCLM-baseline --length 5xC --name DCLM-baseline  
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data DCLM-baseline --length 5xC --name DCLM-baseline  
scripts/beaker/ladder-launch.sh 1 normal --model 530M --data DCLM-baseline --length 5xC --name DCLM-baseline  
scripts/beaker/ladder-launch.sh 1 normal --model 150M --data DCLM-baseline --length 5xC --name DCLM-baseline  
scripts/beaker/ladder-launch.sh 1 normal --model 150M --data dolma17-75p-DCLM-baseline-25p --length 5xC --name dolma17-75p-DCLM-baseline-25p  
scripts/beaker/ladder-launch.sh 1 normal --model 530M --data dolma17-75p-DCLM-baseline-25p --length 5xC --name dolma17-75p-DCLM-baseline-25p  
scripts/beaker/ladder-launch.sh 1 normal --model 300M --data dolma17-75p-DCLM-baseline-25p --length 5xC --name dolma17-75p-DCLM-baseline-25p  
scripts/beaker/ladder-launch.sh 1 normal --model 750M --data dolma17-75p-DCLM-baseline-25p --length 5xC --name dolma17-75p-DCLM-baseline-25p  
scripts/beaker/ladder-launch.sh 1 normal --model 1B --data dolma17-75p-DCLM-baseline-25p --length 5xC --name dolma17-75p-DCLM-baseline-25p   
"""

import argparse
import os
import sys

def main(args):
    with open(args.scales, 'r') as scales_file:
        scales = scales_file.readlines()
    with open(args.data_mixes, 'r') as data_mixes_file:
        data_mixes = data_mixes_file.readlines()
    
    for scale in scales:
        scale = scale.strip()
        for data_mix in data_mixes:
            data_mix = data_mix.strip()
            print(f"scripts/beaker/ladder-launch.sh 1 normal --model {scale} --data {data_mix} --length {args.length} --name {data_mix}-{args.name_suffix}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a ladder of runs over the scales and data mixes provided')
    parser.add_argument('--scales', type=str, help='Path to the file containing the scales')
    parser.add_argument('--data-mixes', type=str, help='Path to the file containing the data mixes')
    parser.add_argument('--name-suffix', type=str, help='suffix (after data-mix) of the experiment')
    parser.add_argument('--length', type=str, help='Length of the experiment')
    args = parser.parse_args()
    main(args)
