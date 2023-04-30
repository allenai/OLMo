"""

Comparing different PII method results.

@kylel

"""

import json
import os

# INDIR = "/Users/kylel/ai2/LLM/pretrain_data/c4/"
# REGEX_OUTPUT_FILE = "part_1017.jsonl__method=regex__postprocess=True__window=100"
# PRESIDIO_OUTPUT_FILE = "part_1017.jsonl__method=presidio__postprocess=True__window=100"

INDIR = "/Users/kylel/ai2/LLM/pretrain_data/common_crawl/"
REGEX_OUTPUT_FILE = "cc_en_tail-0835.json__method=regex__postprocess=True__window=100"
PRESIDIO_OUTPUT_FILE = "cc_en_tail-0835.json__method=presidio__postprocess=True__window=100"

with open(os.path.join(INDIR, REGEX_OUTPUT_FILE)) as f_in:
    regex_preds = [json.loads(line) for line in f_in]
    print(f"Regex rows: {len(regex_preds)}")

with open(os.path.join(INDIR, PRESIDIO_OUTPUT_FILE)) as f_in:
    presidio_preds = [json.loads(line) for line in f_in]
    print(f"Presidio rows: {len(presidio_preds)}")

assert len(regex_preds) == len(presidio_preds)


def reformat_for_display(pii_pred: dict) -> tuple:
    return (pii_pred["score"], [pii[-1] for pii in pii_pred["spans"]])


# Individual PII identification comparison
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
print()

print(f"Presidio caught {len(presidio_only_pii)} that Regex didnt catch")
for p in presidio_only_pii:
    print(f"\t{p}")
print()

print(f"Regex caught {len(regex_only_pii)} that Presidio didnt catch")
for p in regex_only_pii:
    print(f"\t{p}")
print()

print(f"Docs both caught have PII: {len(both_pii)} or {len(both_pii) / len(regex_preds)}")
for p in both_pii:
    print(f"\tPresidio={p['presidio']}")
    print(f"\Regex={p['regex']}")


# Document ranking comparison
sorted_regex_preds = sorted(
    [(i, pred) for i, pred in enumerate(regex_preds)], key=lambda tup: tup[1]["score"], reverse=True
)
sorted_presidio_preds = sorted(
    [(i, pred) for i, pred in enumerate(presidio_preds)], key=lambda tup: tup[1]["score"], reverse=True
)


# loop through top 1, 2, ..., 10% of preds and see how they differ between the two methods
def compare_ranks(pct: int, preview: int = 5):
    n = int(pct / 100 * len(sorted_regex_preds))
    print(f"Top {pct}% of docs is n={n} docs:")
    regex_docs = set([tup[0] for tup in sorted_regex_preds[:n]])
    presidio_docs = set([tup[0] for tup in sorted_presidio_preds[:n]])

    agreement_docs = regex_docs.intersection(presidio_docs)
    print(f"\tBoth agree high PII score: {len(agreement_docs)} or {len(agreement_docs) / n}")
    for id in list(agreement_docs)[:preview]:
        print(f"\t\t{reformat_for_display(pii_pred=regex_preds[id])}")

    regex_only_docs = regex_docs.difference(presidio_docs)
    print(f"\tRegex only: {len(regex_only_docs)} or {len(regex_only_docs) / n}")
    for id in list(regex_only_docs)[:preview]:
        print(f"\t\t{reformat_for_display(pii_pred=regex_preds[id])}")

    presidio_only_docs = presidio_docs.difference(regex_docs)
    print(f"\tPresidio only: {len(presidio_only_docs)} or {len(presidio_only_docs) / n}")
    for id in list(presidio_only_docs)[:preview]:
        print(f"\t\t{reformat_for_display(pii_pred=presidio_preds[id])}")

    print()


compare_ranks(pct=1)
compare_ranks(pct=3)
compare_ranks(pct=5)
compare_ranks(pct=10)
# print(f"\tRegex: {len(regex_docs)}")
# for id in regex_docs:
#     print(f"\t\t{reformat_for_display(pii_pred=regex_preds[id])}")
# print(f"\tPresidio: {len(presidio_docs)}")
# for id in presidio_docs:
#     print(f"\t\t{reformat_for_display(pii_pred=presidio_preds[id])}")
