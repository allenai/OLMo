"""
Multiple streams is not working. So we do this ridiculousness.
"""
import os
import json

def create_config(lang: str, dirpath: str):
    stream = {
            "name": "v2",
            "documents": [
                f"pretraining-data/sources/stack-dedup/v1/documents/{lang}/*.gz",
            ],
            "output": {
                "path": f"pretraining-data/sources/stack-dedup/v2/documents/{lang}",
                "max_size_in_bytes": 1
            },
            "attributes": ["rpj-heuristics"],
            "filter": {
                "include": [

                ],
                "exclude": [
                    "$.attributes.[?(@.rpj_heuristics__code_redpajama_taggers_v1__max_line_length_doc[0][2] > 1000)]",
                    "$.attributes.[?(@.rpj_heuristics__code_redpajama_taggers_v1__avg_line_length_doc[0][2] > 100)]",
                    "$.attributes.[?(@.rpj_heuristics__code_redpajama_taggers_v1__alnum_prop_doc[0][2] < 0.25)]",
                    "$.attributes.[?(@.rpj_heuristics__code_redpajama_taggers_v1__alpha_token_prop_doc < 1.5)]"
                ]
            }
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

