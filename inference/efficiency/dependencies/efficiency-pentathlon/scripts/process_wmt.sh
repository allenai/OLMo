#!/bin/bash

python process_data.py --dataset_path "wmt14" --dataset_name "de-en" --split test --output_folder /home/haop/datasets/

python process_data.py --dataset_path "wmt16" --dataset_name "ro-en" --split test --output_folder /home/haop/datasets/
