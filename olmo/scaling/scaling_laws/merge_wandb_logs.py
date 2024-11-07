import csv
import sys

train_path = sys.argv[1]
eval_path = sys.argv[2]

train_row_by_step = {}
with open(train_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        step = int(row["_step"])
        train_row_by_step[step] = row

rows = []
with open(eval_path, "r") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        step = int(row["_step"])
        if step in train_row_by_step:
            train_row = train_row_by_step[step]
            train_row = {
                k: train_row[k]
                for k in [
                    "throughput/total_tokens",
                    "throughput/total_training_Gflops",
                    "optim/learning_rate_group0",
                ]
            }
            row.update(train_row)
            rows.append(row)

with open(eval_path, "w") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
