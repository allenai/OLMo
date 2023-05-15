import gzip
import json
import logging
import sys
from typing import Dict, Set, Tuple

MIN_WORDS_PER_LINE = 3
NAUGHTY_WORDS: Set[str]
NAUGHTY_PHRASES: Set[str]


def get_text_and_attributes(text: str) -> Tuple[str, Dict]:
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
                attrs["has_naughty_word"] = True
            if any(word == "javascript" for word in words):
                attrs["has_javascript"] = True
            if "lorem ipsum" in line:
                attrs["has_lorem_ipsum"] = True
            if "{" in line:
                attrs["has_curly_brace"] = True
            attrs["line_count"] = len(lines)
            if modified_text:
                modified_text += "\n"
            modified_text += original_line
    except:
        logging.exception(f"Error parsing text: {text}")
        attrs = {}
        modified_text = text

    return modified_text, attrs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python c4.py <input_file> <output_file>")
        sys.exit(1)
    words = open("naughty_words_en.txt").read().splitlines()
    NAUGHTY_WORDS = set(w for w in words if " " not in w)
    NAUGHTY_PHRASES = set(w for w in words if " " in w)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    w = open(output_file, "w")
    for line in gzip.open(input_file):
        obj = json.loads(line)
        text = obj["text"]
        modified_text, attrs = get_text_and_attributes(text)
        out_obj = {
            "id": obj["id"],
            "attributes": attrs,
            "modified_text": modified_text,
        }
        out_json = json.dumps(out_obj)
        w.write(out_json + "\n")
        w.flush()
    w.close()
