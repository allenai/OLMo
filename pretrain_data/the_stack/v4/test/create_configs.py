"""
Multiple streams is not working. So we create multiple configs instead.
"""
import os
import json

def create_config(lang: str, dirpath: str):
    stream = {
            "name": "v4-held-out",
            "documents": [
                f"pretraining-data/sources/stack-dedup/v4/documents/{lang}/*.gz",
            ],
            "output": {
                "path": f"pretraining-data/sources/stack-dedup/v4-held-out/documents/{lang}",
                "max_size_in_bytes": 1
            },
            "attributes": ["basic"],
            "filter": {
                "include": ["$.attributes[?(@.basic__random_number_v1__random[0][2] > 0.996)]"],
                "exclude": []
            },
        }

    config = {
        "streams": [stream],
        "work_dir": {"input": "/tmp/mixer/input", "output": "/tmp/mixer/output"},
        "processes": 32
        }

    
    with open(os.path.join(dirpath, lang + ".json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    import sys

    create_config(sys.argv[1], sys.argv[2])

