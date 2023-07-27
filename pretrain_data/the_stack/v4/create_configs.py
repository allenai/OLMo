"""
Multiple streams is not working. So we do this ridiculousness.
"""
import os
import json

def create_config(lang: str, dirpath: str):
    #if lang in ["java", "python", "javascript"]:
    exclude = [
        "$.attributes.[?(@.starcoder_v2__code_starcoder_taggers_v2__has_xml_template_doc[0][2] > 0.0)]",
        "$.attributes.[?(@.starcoder_v2__code_starcoder_taggers_v2__code_to_comment_ratio_doc[0][2] > 0.8)]",
        "$.attributes.[?(@.starcoder_v2__code_starcoder_taggers_v2__code_to_comment_ratio_doc[0][2] <= 0.01)]",
        "$.attributes.[?(@.starcoder_v2__code_starcoder_taggers_v2__code_to_text_ratio_html_doc[0][2] <= 0.2)]"
    ]
    #else:
    #    exclude = [
    #        "$.attributes.[?(@.starcoder__code_starcoder_taggers_v1__has_xml_template_doc[0][2] > 0.0)]",
    #        "$.attributes.[?(@.starcoder__code_starcoder_taggers_v1__code_to_text_ratio_html_doc[0][2] <= 0.2)]"
    #    ]
    stream = {
            "name": "v4",
            "documents": [
                f"pretraining-data/sources/stack-dedup/v3/documents/{lang}/*.gz",
            ],
            "output": {
                "path": f"pretraining-data/sources/stack-dedup/v4/documents/{lang}",
                "max_size_in_bytes": 1
            },
            "attributes": ["starcoder-v2"],
            "filter": {
                "include": [],
                "exclude": exclude
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

