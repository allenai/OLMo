"""
Script to create perplexity eval datasets for code.
"""
import os

import pandas as pd
from datasets import Dataset, load_dataset


def create_raw_dataset(dataset: Dataset, id_key: str, prompt_key: str, answer_key: str, save_to: str):
    instances = []
    for instance in dataset:
        updated_instance = {}
        updated_instance["id"] = instance.pop(id_key)
        updated_instance["text"] = instance.pop(prompt_key) + instance.pop(answer_key)
        updated_instance["metadata"] = instance

        instances.append(updated_instance)

    df = pd.DataFrame.from_records(instances)
    df.to_json(save_to, lines=True, compression="gzip", orient="records")


def run(outdir: str):
    # Dataset: openai_humaneval
    datadict = load_dataset("openai_humaneval")
    create_raw_dataset(
        datadict["test"],
        id_key="task_id",
        prompt_key="prompt",
        answer_key="canonical_solution",
        save_to=os.path.join(outdir, "openai_humaneval_test.jsonl.gz"),
    )

    # Dataset: mbpp
    datadict = load_dataset("mbpp")  # full dataset
    create_raw_dataset(
        datadict["validation"],
        id_key="task_id",
        prompt_key="text",
        answer_key="code",
        save_to=os.path.join(outdir, "mbpp_validation.jsonl.gz"),
    )
    create_raw_dataset(
        datadict["test"],
        id_key="task_id",
        prompt_key="text",
        answer_key="code",
        save_to=os.path.join(outdir, "mbpp_test.jsonl.gz"),
    )


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
