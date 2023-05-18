"""

Evaluating different thresholds.

@kylel

"""

import json
import os

INDIR = "/Users/kylel/ai2/LLM/pretrain_data/common_crawl/"
HATESPEECH_OUTPUT_FILE = "cc_en_tail-0835.json__model=hatespeech__level=sent__threshold=0.4"

with open(os.path.join(INDIR, HATESPEECH_OUTPUT_FILE)) as f_in:
    hatespeech_preds = [json.loads(line) for line in f_in]
    print(f"Hatespeech rows: {len(hatespeech_preds)}")


def reformat_for_display(hatespeech_pred: dict) -> tuple:
    return (hatespeech_pred["score"], [pred[-1] for pred in hatespeech_pred["spans"]])


# What is yield?
print(f"Total number of docs: {len(hatespeech_preds)}")
print(f"Total number of spans: {sum([len(pred['spans']) for pred in hatespeech_preds])}")
print(f"Total number of docs with at least one span: {len([pred for pred in hatespeech_preds if pred['spans']])}")
print(f"Average number of spans in docs with at least one span: {sum([len(pred['spans']) for pred in hatespeech_preds if pred['spans']]) / len([pred for pred in hatespeech_preds if pred['spans']])}")


# restrict to docs with at least one span
hatespeech_preds = [pred for pred in hatespeech_preds if pred["spans"]]


# sort them by number of spans detected
sorted_hatespeech_preds = sorted(hatespeech_preds, key=lambda x: len(x["spans"]), reverse=True)

for pred in sorted_hatespeech_preds[:10]:
    print(reformat_for_display(pred))