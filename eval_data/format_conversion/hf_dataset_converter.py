import argparse
import datetime
import gzip
import json
import os
from uuid import uuid4

from datasets import load_dataset

"""
Sample script to convert all the datasets into the required format. 

for dataset in arc_challenge_test arc_easy_test boolq_dev hellaswag_test obqa_test mmlu_test \
    xsum_test imdb_test scitldr_test multi_eurlex_test drop_val headqa_test piqa_test \
    winogrande_test logiqa_test;
do 
  d
          --dataset $dataset \
          --out_dir outputs/downstream_perplexity/
done
"""


class HFFormatter:
    @staticmethod
    def arc_challenge_test(args):
        dataset = load_dataset("ai2_arc", "ARC-Challenge")
        for item in dataset["test"]:
            yield {
                "id": item["id"],
                "text": item["question"],
            }

    @staticmethod
    def arc_easy_test(args):
        dataset = load_dataset("ai2_arc", "ARC-Easy")
        for item in dataset["test"]:
            yield {
                "id": item["id"],
                "text": item["question"],
            }

    @staticmethod
    def boolq_dev(args):
        # Note only validation set is available
        dataset = load_dataset("boolq")
        for item in dataset["validation"]:
            # No id field
            yield {
                "text": item["question"],
            }

    @staticmethod
    def hellaswag_test(args):
        # Note only validation set is available
        dataset = load_dataset("hellaswag")
        for item in dataset["test"]:
            yield {
                "id": item["ind"],
                # Use the first sentence of the context for contamination check
                "text": item["ctx_a"],
            }

    @staticmethod
    def obqa_test(args):
        dataset = load_dataset("openbookqa", "main")
        for item in dataset["test"]:
            yield {
                "id": item["id"],
                "text": item["question_stem"],
            }

    @staticmethod
    def mmlu_test(args):
        # Create a single file from all sub-datasets. It doesn't seem necessary to separate each
        # sub-dataset currently.
        mmlu_configs = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
                        'clinical_knowledge', 'college_biology', 'college_chemistry',
                        'college_computer_science', 'college_mathematics', 'college_medicine',
                        'college_physics', 'computer_security', 'conceptual_physics',
                        'econometrics', 'electrical_engineering', 'elementary_mathematics',
                        'formal_logic', 'global_facts', 'high_school_biology',
                        'high_school_chemistry', 'high_school_computer_science',
                        'high_school_european_history', 'high_school_geography',
                        'high_school_government_and_politics', 'high_school_macroeconomics',
                        'high_school_mathematics', 'high_school_microeconomics',
                        'high_school_physics', 'high_school_psychology', 'high_school_statistics',
                        'high_school_us_history', 'high_school_world_history', 'human_aging',
                        'human_sexuality', 'international_law', 'jurisprudence',
                        'logical_fallacies', 'machine_learning', 'management', 'marketing',
                        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
                        'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
                        'professional_law', 'professional_medicine', 'professional_psychology',
                        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
                        'virology', 'world_religions']
        for config in mmlu_configs:
            # Load this particular dataset since it does not include the unnecessary and
            # extremely large train splits
            dataset = load_dataset("tasksource/mmlu", config)
            for item in dataset["test"]:
                yield {
                    "text": item["question"],
                }

    @staticmethod
    def xsum_test(args):
        dataset = load_dataset("xsum")
        for item in dataset["test"]:
            yield {
                "id": item["id"],
                "text": item["summary"],
            }

    @staticmethod
    def imdb_test(args):
        dataset = load_dataset("imdb")
        for item in dataset["test"]:
            yield {
                "text": item["text"],
            }

    @staticmethod
    def scitldr_test(args):
        # Same target text for all splits, so using the smallest one: Abstract
        dataset = load_dataset("allenai/scitldr", "Abstract")
        for item in dataset["test"]:
            for target_text in item["target"]:
                yield {
                    "text": target_text
                }

    @staticmethod
    def multi_eurlex_test(args):
        # Most likely contamination on the en split
        dataset = load_dataset("multi_eurlex", "en")
        for item in dataset["test"]:
            yield {
                # look for source text
                "text": item["text"],
            }

    @staticmethod
    def drop_val(args):
        # Most likely contamination on the en split
        dataset = load_dataset("drop")
        for item in dataset["validation"]:
            yield {
                "text": item["question"],
            }

    @staticmethod
    def headqa_test(args):
        # Most likely contamination on the en split
        dataset = load_dataset("head_qa", "en")
        for item in dataset["test"]:
            yield {
                "text": item["qtext"],
            }

    @staticmethod
    def piqa_test(args):
        # Note only validation set is available
        dataset = load_dataset("piqa")
        for item in dataset["test"]:
            yield {
                "text": item["goal"],
            }

    @staticmethod
    def winogrande_test(args):
        # Note that the test set does not change between the smallest or largest dataset?
        dataset = load_dataset("winogrande", "winogrande_xs")
        for item in dataset["test"]:
            yield {
                "text": item["sentence"],
            }

    @staticmethod
    def logiqa_test(args):
        # Note that the test set does not change between the smallest or largest dataset?
        dataset = load_dataset("lucasmccabe/logiqa")
        for item in dataset["test"]:
            yield {
                "text": item["context"],
            }


def main():
    parse = argparse.ArgumentParser("")

    parse.add_argument("--dataset", type=str,
                       help="Dataset Name (must have corresponding HFFormatter method)")
    parse.add_argument("--out_dir", type=str, help="Output Directory")

    args = parse.parse_args()
    data = getattr(HFFormatter, args.dataset)(args)
    with gzip.open(os.path.join(args.out_dir, args.dataset + '.jsonl.gz'), 'wt') as fout:
        for doc in data:
            doc['id'] = str(uuid4()) if 'id' not in doc else doc['id']
            doc['added'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc['source'] = args.dataset if 'source' not in doc else doc['source']
            fout.write(json.dumps(doc) + '\n')


if __name__ == "__main__":
    main()
