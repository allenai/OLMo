import numpy as np
import json
import re
import boto3

from olmo.data.named_data_mixes import DATA_SOURCES

s3 = boto3.client("s3")

avg_size_per_source = {}
# take paths from each DATA_SOURCES key. Compute size of file with boto3.
for key, paths in DATA_SOURCES.items():
    elements = []
    for file in paths:
        response = s3.head_object(Bucket="ai2-llm", Key=file)
        num_elements = response['ContentLength'] // np.dtype(np.uint16).itemsize
        elements.append(num_elements)
    num_files = len(paths)
    avg_size_per_source[key] = {
        "avg_file_size": int(np.mean(elements)),
        "num_files": num_files,
        "total_size": int(np.sum(elements)),
    }
    print(f"Stats for {key}: {avg_size_per_source[key]}")

txt = json.dumps(avg_size_per_source, indent=4)
# for every number, add _ to separate thousands
txt = re.sub(r'(\d)(?=(\d{3})+(?!\d))', r'\1_', txt)

print(txt)
