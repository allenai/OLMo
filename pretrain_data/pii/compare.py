"""

Comparing different PII method results.

@kylel

"""

import json
import os

# INDIR = "/Users/kylel/ai2/LLM/pretrain_data/c4/"
# REGEX_OUTPUT_FILE = "part_1017.jsonl__method=regex__postprocess=True__window=50"
# PRESIDIO_OUTPUT_FILE = "part_1017.jsonl__method=presidio__postprocess=True__window=50"
INDIR = "/Users/kylel/ai2/LLM/pretrain_data/common_crawl/"
REGEX_OUTPUT_FILE = "cc_en_head-1334.json__method=regex__postprocess=True__window=50"
PRESIDIO_OUTPUT_FILE = "cc_en_head-1334.json__method=presidio__postprocess=True__window=50"

with open(os.path.join(INDIR, REGEX_OUTPUT_FILE)) as f_in:
    regex_preds = [json.loads(line) for line in f_in]
    print(f"Regex rows: {len(regex_preds)}")

with open(os.path.join(INDIR, PRESIDIO_OUTPUT_FILE)) as f_in:
    presidio_preds = [json.loads(line) for line in f_in]
    print(f"Presidio rows: {len(presidio_preds)}")

assert len(regex_preds) == len(presidio_preds)


def reformat_for_display(pii_pred: dict) -> tuple:
    return (pii_pred["score"], [pii[-1] for pii in pii_pred["pii"]])


both_no_pii = 0
regex_only_pii = []
presidio_only_pii = []
both_pii = []
for i in range(len(regex_preds)):
    regex_pred = regex_preds[i]
    presidio_pred = presidio_preds[i]
    if regex_pred["score"] == 0.0 and presidio_pred["score"] == 0.0:
        both_no_pii += 1
    elif regex_pred["score"] == 0.0:
        presidio_only_pii.append(reformat_for_display(pii_pred=presidio_pred))
    elif presidio_pred["score"] == 0.0:
        regex_only_pii.append(reformat_for_display(pii_pred=regex_pred))
    else:
        both_pii.append(
            {
                # just display the mention
                "presidio": reformat_for_display(pii_pred=presidio_pred),
                "regex": reformat_for_display(pii_pred=regex_pred),
            }
        )

print(f"Both agree no PII: {both_no_pii} or {both_no_pii / len(regex_preds)}")

print(f"Presidio caught {len(presidio_only_pii)} that Regex didnt catch")
for p in presidio_only_pii:
    print(f"\t{p}")

print(f"Regex caught {len(regex_only_pii)} that Presidio didnt catch")
for p in regex_only_pii:
    print(f"\t{p}")

print(f"Docs both caught have PII: {len(both_pii)} or {len(both_pii) / len(regex_preds)}")
for p in both_pii:
    print(f"\tPresidio={p['presidio']}")
    print(f"\Regex={p['regex']}")
