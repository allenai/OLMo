"""Preprocess data for efficiency benchmark."""

import argparse

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset

MINIMUM_SINGLE_STREAM_INSTANCES = 1000
NUM_OFFLINE_BATCHES = 10000
OFFLINE_BATCH_SIZE = 32


def write_accuracy_data(dataset: Dataset, output_folder: str):
    """Data for accuracy evaluation."""
    dataset.to_json(f"{output_folder}/accuracy.json")


def write_single_stream_data(dataset: Dataset, output_folder: str):
    """Data for single stream scenario efficiency evaluation."""
    single_stream_instances = dataset.to_list()
    while len(single_stream_instances) < MINIMUM_SINGLE_STREAM_INSTANCES:
        single_stream_instances.extend(dataset.to_list())
    single_stream_dataset = Dataset.from_pandas(pd.DataFrame(single_stream_instances))
    single_stream_dataset.to_json(f"{output_folder}/single_stream.json")


def write_offline_data(dataset: Dataset, output_folder: str):
    """Data for offline scenario efficiency evaluation."""
    instances = dataset.to_list()
    num_instances = len(dataset)
    batch_size = min(num_instances, OFFLINE_BATCH_SIZE)
    batches = []
    for i in range(NUM_OFFLINE_BATCHES):
        batch = np.random.choice(instances, size=batch_size)
        batches.extend(list(batch))
    offline_dataset = Dataset.from_pandas(pd.DataFrame(batches))
    offline_dataset.to_json(f"{output_folder}/offline.json")


if __name__ == "__main__":
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.dataset_path, args.dataset_name, split=args.split)

    write_accuracy_data(dataset, output_folder=f"{args.output_folder}/{args.dataset_path}/{args.dataset_name}")

    write_single_stream_data(
        dataset, output_folder=f"{args.output_folder}/{args.dataset_path}/{args.dataset_name}"
    )

    write_offline_data(dataset, output_folder=f"{args.output_folder}/{args.dataset_path}/{args.dataset_name}")
