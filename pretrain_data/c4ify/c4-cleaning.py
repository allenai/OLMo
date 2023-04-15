# The utility code for C4 preprocessing logics.
# The file is inspired by the following code:
# https://github.com/shjwudp/c4-dataset-script/tree/master
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/c4_utils.py

import codecs
import hashlib
import heapq
import json
import os
import re
from typing import Any, Dict, Optional, Sequence, TypedDict

import fire  # type: ignore
import nltk  # type: ignore
import requests  # type: ignore
import six as _six  # type: ignore


# To keep our naming consistent with the original C4 dataset,
# We call each individual document a "page".
# But instead of converting it to a dataclass, we just use dict
# and specify the data with a TypedDict
class Page(TypedDict):
    url: str
    normalized_url: str
    text: str
    timestamp: str
    content_length: str
    content_type: str
    language: Optional[str]
    language_score: Optional[float]


## Constants # noqa: E266
# refer to https://github.com/tensorflow/datasets/blob/daf616684ea224c445e331f53cd5d0f7877477f7/tensorflow_datasets/text/c4_utils.py#L41
_MIN_WORDS_PER_LINE = 5
_MIN_NUM_SENTENCES = 3
_MAX_WORD_LENGTH = 1000

_END_MARKS = (".", "?", "!", '"')
_ELLIPSIS = "..."
_POLICY_SUBSTRINGS = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
]
_CITATION_REGEX = re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")

# fmt: off
_EN_BADWORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/4638b970cb8d9d82789564fcba1f4a1eb508ff1a/en"
# TODO: This link is slightly different from the original C4 link because
# there has been new commits to the original repo.
_BADWORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/5faf2ba42d7b1c0977169ec3611df25a3c08eb13/{lang}"
_BADWORDS_LANGS = [
    "ar", "cs", "da", "de", "en", "eo", "es", "fa", "fi", "fil", "fr",
    "fr-CA-u-sd-caqc", "hi", "hu", "it", "ja", "kab", "ko", "nl", "no",
    "pl", "pt", "ru", "sv", "th", "tlh", "tr", "zh",
]
_BADWORDS_CACHE_DIR = "badwords_cache"
_BADWORDS_CACHE_FILE = "badwords.json"
# fmt: on


## General utility functions # noqa: E266


# This is the as_text function from tensorflow.compat.as_text
# https://github.com/tensorflow/tensorflow/blob/0db597d0d758aba578783b5bf46c889700a45085/tensorflow/python/util/compat.py#L89-L112
def as_text(bytes_or_text, encoding="utf-8"):
    """Converts any string-like python input types to unicode.
    Returns the input as a unicode string. Uses utf-8 encoding for text
    by default.
    Args:
      bytes_or_text: A `bytes`, `str`, or `unicode` object.
      encoding: A string indicating the charset for decoding unicode.
    Returns:
      A `unicode` (Python 2) or `str` (Python 3) object.
    Raises:
      TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    # Validate encoding, a LookupError will be raised if invalid.
    encoding = codecs.lookup(encoding).name
    if isinstance(bytes_or_text, _six.text_type):
        return bytes_or_text
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text.decode(encoding)
    else:
        raise TypeError("Expected binary or unicode string, got %r" % bytes_or_text)


def normalize_url(url):
    url = as_text(url)
    url = re.sub(r"https?:\/\/(www\.)?", "", url)
    url = re.sub(r"\?(utm_|ref|feed).*", "", url)
    url = url.rstrip("/")
    return url


_SENTENCE_TOKENIZER = None


def _load_sentence_tokenizer():
    """Returns a sentence tokenization function."""
    # Lock to avoid a race-condition in the creation of the download directory.
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def get_sentences(text):
    global _SENTENCE_TOKENIZER
    if not _SENTENCE_TOKENIZER:
        _SENTENCE_TOKENIZER = _load_sentence_tokenizer()
    return list(_SENTENCE_TOKENIZER.tokenize(as_text(text)))


def cache_and_load_badwords() -> Dict[str, Sequence[str]]:
    if os.path.exists(_BADWORDS_CACHE_FILE):
        with open(_BADWORDS_CACHE_FILE, "r", encoding="utf-8") as file:
            badwords_dict = json.load(file)
            return {key: list(item) for key, item in badwords_dict.items()}

    # Otherwise, download the badwords and cache them
    badwords_dict = {}
    if not os.path.exists(_BADWORDS_CACHE_DIR):
        os.makedirs(_BADWORDS_CACHE_DIR)

    for lang in _BADWORDS_LANGS:
        local_file = os.path.join(_BADWORDS_CACHE_DIR, f"{lang}.txt")

        if not os.path.exists(local_file):
            url = _BADWORDS_URL.format(lang=lang)
            response = requests.get(url)

            if response.status_code == 200:
                with open(local_file, "w", encoding="utf-8") as file:
                    file.write(response.text)
            else:
                print(f"Failed to download bad words for language: {lang}")

        with open(local_file, "r", encoding="utf-8") as file:
            badwords_dict[lang] = set(word.strip() for word in file.readlines())

    with open(_BADWORDS_CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump({key: list(item) for key, item in badwords_dict.items()}, file)

    return badwords_dict


def load_badwords_regex():
    """Returns a set of badwords for a given language."""
    badwords = cache_and_load_badwords()
    badwords_regex = {}
    for lang, words in badwords.items():
        words = [re.escape(w) for w in words]
        badwords_regex[lang] = (
            # For Japanese, Thai, and Chinese, do not require word separations.
            re.compile("|".join(words))
            if lang in ("ja", "th", "zh")
            # For other languages, match only when flanked by non-word chars.
            else re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
        )
    return badwords_regex


def get_hashed_url_filter_fn(predicate_fn):
    def filter_fn(page):
        url = page.normalized_url
        val = int(hashlib.md5(as_text(url).encode("utf-8")).hexdigest(), 16)
        return predicate_fn(val)

    return filter_fn


## Page filtering functions # noqa: E266

# conventions for filtering functions:
# - return True if the page passes the filter
# - return False if the page fails the filter


def page_filter_by_paragraphs(page: Page, min_paragraphs=3, min_paragraph_len=200, line_delimiter="\n") -> bool:
    """Returns False iff a page has too few or too short paragraphs."""
    lines = page["text"].split(line_delimiter)
    # Filter out docs that don't have at least three "paragraphs"
    # (lines >= `min_paragraph_len` chars).
    if len(lines) < min_paragraphs or min(heapq.nlargest(3, [len(line) for line in lines])) < min_paragraph_len:
        return False
    return True


def page_filter_by_valid_length(page: Page, max_length=1.9e5) -> bool:
    """Returns False iff page's text is too long."""
    if len(page["text"]) > max_length:
        return False
    return True


def page_filter_by_language_and_score(page: Page, language="en", min_probability=0.99) -> bool:
    """Returns False iff page's language is not English."""
    if page["language"] != language and page["language_score"] < min_probability:
        return False
    return True


def page_processing_by_lines(
    page: Page,
    citation_regex: re.Pattern = _CITATION_REGEX,
    min_words_per_line=_MIN_WORDS_PER_LINE,
    min_num_sentences=_MIN_NUM_SENTENCES,
    max_word_length=_MAX_WORD_LENGTH,
) -> Optional[Page]:
    """
    The main function for processing a pag content.
    Returns a new page with the text field processed, or None if the page is
    discarded when failing the filtering criteria.
    """

    def line_has_too_long_word(line):
        for word in line.split():
            if len(word) > max_word_length:
                return True
        return False

    lines = page["text"].splitlines()

    valid_lines = []
    num_sentences = 0

    for line in lines:
        # 1. [processing] strip whitespace
        line = line.strip()

        # 2. [filtering] skip long lines
        if line_has_too_long_word(line):
            continue

        # 3. [processing] remove citations
        line = citation_regex.sub("", line)

        # 4. [filtering] skip lines that don't end with a sentence mark
        if not line.endswith(_END_MARKS) or line.endswith(_ELLIPSIS):
            continue

        # 5. [filtering] skip lines with too few words
        if len(line.split()) < min_words_per_line:
            continue

        line_lower = line.lower()

        # 6. [filtering] skip lines with lorem ipsum
        if "lorem ipsum" in line_lower:
            return None

        # 7. [filtering] skip lines with "javascript must be enabled" notices
        # TODO: originally it is 'javascript' in line_lower
        # but perhaps we might want to switch to 'javascript must be enabled'
        # https://github.com/tensorflow/datasets/blob/daf616684ea224c445e331f53cd5d0f7877477f7/tensorflow_datasets/text/c4_utils.py#L259
        if "javascript must be enabled" in line_lower:
            continue

        # 8. [filtering] Remove docs which probably contain javascript code
        # TODO: though I am suspicious of this line and dropped them
        # if "{" in line:
        #     return

        # 9. [filtering] Remove policy lines
        if any(p in line_lower for p in _POLICY_SUBSTRINGS):
            continue

        num_sentences += len(get_sentences(line))
        valid_lines.append(line)

    # 10. [filtering] skip docs with too few sentences
    if num_sentences < min_num_sentences:
        return None

    page["text"] = "\n".join(valid_lines).strip()
    return page


def page_filter_by_badwords(page, badwords_regex: Dict[str, Sequence[re.Pattern]], filter_fraction: float = 1.0):
    """
    Filters pages at given rate that contain language-specific bad word(s).
    When we detect a bad word on a page, we drop them with a chance of
    `filter_fraction`. This is done by hashing the page's URL and dropping the
    page if the hash is less than `filter_fraction`. As such, this is a deterministic
    filter, and the same page will always be dropped or kept.
    """

    filter_ratio = float.as_integer_ratio(filter_fraction)
    keep_badword_page = get_hashed_url_filter_fn(lambda x: x % filter_ratio[1] >= filter_ratio[0])

    lang = page["language"].split("-")[0]  # remove suffix if present
    if lang in badwords_regex:
        text = page["text"]
        badwords_found = badwords_regex[lang].search(text.lower())  # type: ignore
        if badwords_found is not None:
            if keep_badword_page(page):
                return True
            return False
    return True


## Load from source data # noqa: E266


def convert_original_data_to_page(data: Dict[str, Any]) -> Page:
    return {
        "url": data["metadata"]["url"],
        "normalized_url": normalize_url(data["metadata"]["url"]),
        "text": data["text"],
        "timestamp": data["created"],
        "content_length": data["metadata"]["length"],
        "content_type": "",
        "language": data["metadata"]["language"],
        "language_score": data["metadata"]["language_score"],
    }


def load_and_parse_file(
    file_path: str,
    save_path: str,
    max_lines: Optional[int] = None,
):
    all_data = []
    with open(file_path, "r") as fp:
        for line in fp.readlines(max_lines):  # type: ignore
            all_data.append(json.loads(line))

    badwords = load_badwords_regex()
    all_pages_to_save = []

    for data in all_data:
        page = convert_original_data_to_page(data)

        if not page_filter_by_language_and_score(page, 0.99):
            continue

        if not page_filter_by_valid_length(page):
            continue

        if not page_filter_by_paragraphs(page):
            continue

        converted_page = page_processing_by_lines(page)
        if converted_page is None:
            continue

        if not page_filter_by_badwords(converted_page, badwords):
            continue

        all_pages_to_save.append(page)

    with open(save_path, "w") as fp:
        for page in all_pages_to_save:
            fp.write(json.dumps(page) + "\n")


if __name__ == "__main__":
    fire.Fire(load_and_parse_file)
