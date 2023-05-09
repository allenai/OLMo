import argparse
import logging
import re
import string
import sys
from typing import Dict, Union

from pretrain_data.the_stack.create_utils import (
    _get_lang_list,
    create_attributes,
    create_documents,
    should_exclude_filename,
)
from uniseg.wordbreak import words as unicode_tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("_create_v1.log"),
    ],
)

logger = logging.getLogger(__name__)


def clean_copyright_comments(content: str):
    # Regex to strip repeated copyright comment blocks
    CPAT = re.compile("copyright", re.IGNORECASE)
    PAT = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")

    r = PAT.search(content)
    if r:
        # found one, now see if it contains "copyright", if so strip it
        span = r.span()
        sub = content[span[0] : span[1]]
        if CPAT.search(sub):
            # cut it
            content = content[: span[0]] + content[span[1] :]

        return content

    lines = content.split("\n")
    skip = 0

    # Greedy replace any file that begins with comment block, most
    # are copyright headers
    for k in range(len(lines)):
        if lines[k].startswith("//") or lines[k].startswith("#") or lines[k].startswith("--") or not lines[k]:
            skip = skip + 1
        else:
            break

    if skip:
        # we skipped, consume it
        content = "\n".join(lines[skip:])

    return content


def count_tokens_unicode(text):
    # this is extremely slow
    count = sum(1 for word in unicode_tokenize(text) if not all(char in string.whitespace for char in word))
    return count


def get_filecontent_stats(instance, clean_copyright: bool = False) -> Dict[str, Union[int, str]]:
    # split content into lines and get line lengths
    content = instance["text"]
    if clean_copyright:
        content = clean_copyright_comments(content)

    line_lengths = list(map(len, content.splitlines()))

    if len(line_lengths) == 0:
        instance.update(
            {
                "line_count": 0,
                "max_line_length": 0,
                "avg_line_length": 0,
                "alnum_count": 0,
                "alnum_prop": 0,
                "alpha_count": 0,
                "num_characters": 0,
                "num_tokens_whitespace": 0,
                "num_alpha": 0,
            }
        )
        return instance

    num_characters = len(content)

    # get max line length
    max_length = max(line_lengths)

    # get average line length
    avg_length = num_characters / len(line_lengths)

    num_tokens_whitespace = len(content.split())

    # get proportion of alphanumeric characters
    alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, content))
    alnum_prop = alnum_count / num_characters

    alpha_count = sum(map(lambda char: 1 if char.isalpha() else 0, content))
    # alpha_token_prop = alpha_count / num_tokens_whitespace

    instance["line_count"] = len(line_lengths)
    instance["max_line_length"] = max_length
    instance["avg_line_length"] = avg_length

    instance["alnum_count"] = alnum_count
    instance["alnum_prop"] = alnum_prop

    instance["alpha_count"] = alpha_count

    instance["num_characters"] = num_characters
    instance["num_tokens_unicode"] = count_tokens_unicode(content) # nobody got time for that

    # whitespace
    instance["num_tokens_whitespace"] = num_tokens_whitespace

    return instance


def process_file(old_version: str, new_version: str, lang_list_path: str, filename: str):
    lang_list = _get_lang_list(lang_list_path)
    if should_exclude_filename(filename, lang_list):
        return

    create_documents(old_version, new_version, filename, [clean_copyright_comments])
    create_attributes(old_version, new_version, filename, [get_filecontent_stats])
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new version files from corresponding old version files.")
    parser.add_argument("--old-version", type=str, required=False, default="v0")
    parser.add_argument("--new-version", type=str, required=False, default="v1")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--lang-list", type=str, required=False, default="lang_list.txt")
    args = parser.parse_args()

    process_file(args.old_version, args.new_version, args.lang_list, args.filename)
