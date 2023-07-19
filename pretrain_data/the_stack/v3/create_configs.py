"""
Multiple streams is not working. So we do this ridiculousness.
"""
import os
import json

def create_config(lang: str, dirpath: str):
    stream = {
            "name": "v3",
            "documents": [
                f"pretraining-data/sources/stack-dedup/v2/documents/{lang}/*.gz",
            ],
            "output": {
                "path": f"pretraining-data/sources/stack-dedup/v3/documents/{lang}",
                "max_size_in_bytes": 1
            },
            "attributes": ["pii"],
            "filter": {
                "include": [],
                "exclude": []
            },
            "span_replacement": [
                {
                    "span": "$.attributes.pii__pii_regex_with_counts_fast_v2__EMAIL_ADDRESS",
                    "min_score": 0.5,
                    "replacement": " |||EMAIL_ADDRESS||| "
                },
                {
                    "span": "$.attributes.pii__pii_regex_with_counts_fast_v2__PHONE_NUMBER",
                    "min_score": 0.5,
                    "replacement": " |||PHONE_NUMBER||| "
                },
                {
                    "span": "$.attributes.pii__pii_regex_with_counts_fast_v2__IP_ADDRESS",
                    "min_score": 0.5,
                    "replacement": " |||IP_ADDRESS||| "
                }
            ]
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

