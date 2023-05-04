import gzip
import json
import logging
import sys
from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Tuple

REQUIRED_ENGLISH_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
SYMBOLS = {"#", "\u2026"}
BULLET_POINTS = {"*", "-"}


def get_attributes(text: str) -> Dict:
    attrs = {}

    try:
        words = text.split()
        word_count = len(words)
        character_count = sum(len(word) for word in words)

        attrs["word_count"] = word_count
        attrs["median_word_length"] = median([len(word) for word in words])
        attrs["symbol_to_word_ratio"] = sum(1 for word in words if any(s in word for s in SYMBOLS)) / word_count
        attrs["fraction_of_words_with_alpha_character"] = (
            sum(1 for word in words if any(c.isalpha() for c in word)) / word_count
        )
        attrs["required_word_count"] = sum(1 for word in words if word in REQUIRED_ENGLISH_WORDS)

        for n in range(2, 5):
            value = 0.0
            ngrams = find_ngrams(words, n)
            if ngrams:
                ngram_counts = occurrence_counts(ngrams)
                most_common_ngram, count = max(ngram_counts.items(), key=lambda x: x[1])
                value = sum(len(s) for s in most_common_ngram) / character_count
            attrs[f"fraction_of_characters_in_most_common_{n}gram"] = value

        for n in range(5, 11):
            value = 0.0
            ngrams = find_ngrams(words, n)
            if ngrams:
                ngram_counts = occurrence_counts(ngrams)
                # Over-count the characters in the same way for the denominator and numerator
                ng_char_count = sum(sum(len(w) for w in ng) for ng in ngrams)
                value = (
                    sum(sum(len(w) for w in ng) * count for ng, count in ngram_counts.items() if count > 1)
                    / ng_char_count
                )
            attrs[f"fraction_of_characters_in_duplicate_{n}grams"] = value

        lines = text.split("\n")
        line_count = len(lines)

        attrs["fraction_of_lines_starting_with_bullet_point"] = (
            sum(1 for line in lines if any(line.startswith(s) for s in BULLET_POINTS)) / line_count
        )
        attrs["fraction_of_lines_ending_with_ellipsis"] = sum(
            1 for line in lines if line.endswith("\u2026")
        ) / len(lines)
        attrs["fraction_of_duplicate_lines"] = (
            sum(count for line, count in occurrence_counts(lines).items() if count > 1) / line_count
        )
        attrs["fraction__of_characters_in_duplicate_lines"] = (
            sum(len(line) * count for line, count in occurrence_counts(lines).items() if count > 1)
            / character_count
        )
    except Exception as e:
        logging.exception(f"Error processing text: {text}")
        attrs = {}

    return attrs


def occurrence_counts(objects: List[Any]) -> Dict[Any, int]:
    counts = defaultdict(int)
    for line in objects:
        counts[line] += 1
    return counts


def find_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    return list(zip(*[words[i:] for i in range(n)]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gopher.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    w = open(output_file, "w")
    for line in gzip.open(input_file):
        obj = json.loads(line)
        text = obj["text"]
        attrs = get_attributes(text)
        out_obj = {"id": obj["id"], "attributes": attrs}
        out_json = json.dumps(out_obj)
        w.write(out_json + "\n")
        w.flush()
    w.close()
