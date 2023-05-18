import gzip
import json
import logging
import os
import sys
from typing import Dict, Set, Tuple

MIN_WORDS_PER_LINE = 3
words_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "naughty_words_en.txt")
words = open(words_file).read().splitlines()
NAUGHTY_WORDS: Set[str] = set(w for w in words if " " not in w)
NAUGHTY_PHRASES: Set[str] = set(w for w in words if " " in w)


def get_attributes(text: str) -> Dict:
    attrs = {}
    modified_text = ""
    try:
        lines = text.split("\n")
        for original_line in lines:
            line = original_line.lower().strip()
            if not line.endswith((".", "?", "!", '"')):
                continue
            words = line.split()
            if len(words) < MIN_WORDS_PER_LINE:
                continue
            if any(word in NAUGHTY_WORDS for word in words) or any(phrase in line for phrase in NAUGHTY_PHRASES):
                attrs["c4_has_naughty_word"] = True
            if any(word == "javascript" for word in words):
                attrs["c4_has_javascript"] = True
            if "lorem ipsum" in line:
                attrs["c4_has_lorem_ipsum"] = True
            if "{" in line:
                attrs["c4_has_curly_brace"] = True
            attrs["c4_line_count"] = len(lines)
            if modified_text:
                modified_text += "\n"
            modified_text += original_line
    except:
        logging.exception(f"Error parsing text: {text}")
        attrs = {}

    attrs["c4_modified_text"] = modified_text
    return attrs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python c4.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    w = open(output_file, "w")
    for line in gzip.open(input_file):
        obj = json.loads(line)
        text = obj["text"]
        attrs = get_attributes(text)
        out_obj = {
            "id": obj["id"],
            "attributes": attrs,
        }
        out_json = json.dumps(out_obj)
        w.write(out_json + "\n")
        w.flush()
    w.close()
