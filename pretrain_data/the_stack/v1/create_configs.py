"""
Multiple streams is not working. So we do this ridiculousness.
"""
import os
import json

def create_config(lang: str, dirpath: str):
    stream = {
            "name": "v1",
            "documents": [
                f"pretraining-data/sources/stack-dedup/v0/documents/{lang}/*.gz",
            ],
            "output": {
                "path": f"pretraining-data/sources/stack-dedup/v1/documents/{lang}",
                "max_size_in_bytes": 1
            },
            "attributes": ["copyright"],
            "span_replacement": [
                {
                    "span": "$.attributes.copyright__code_copyright_comments_v1__comment_block",
                    "min_score": 1.0,
                    "replacement": ""
                },
                {
                    "span": "$.attributes.copyright__code_copyright_comments_v1__copyright_notice",
                    "min_score": 1.0,
                    "replacement": ""
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

