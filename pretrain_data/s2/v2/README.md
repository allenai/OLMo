# S2 V2

## S2ORC (Full-text Papers)

Cleaned version of the S2ORC corpus, which contains full-text papers across multiple disciplines.
Unflitered, the corpus contains 11.3M papers and 46.9B whitespace-separated tokens.
To clean S2ORC, we impose the following constraints:

- The paper must have a title and abstract.
- From each paper, we use [Grobid](https://github.com/kermitt2/grobid) to extract section headers and paragraphs; figures, tables, and references, and any other non-textual content is removed. Title and abstracts are also available, but they come from the S2 metadata, not Grobid.
- The paper must be in English.
  - To calculate the language, we use the [pycld3](https://github.com/bsolomon1124/pycld3) library
  - We run pycld3 on the first 2000 characters of each paragraph in the paper.
  - The language of the paper is the most common language of the paragraphs.
- The paper must have at least 500 words.
- The paper was published after 1969.
- The paper must have at least 5 paragraphs.
  - All sections that have a average log word probability of less than `-20` are removed.
  - To calculate the average log word probability, we use word frequencies extracted from the [1T Web Ngram corpus](https://catalog.ldc.upenn.edu/LDC2006T13); specifically, we use the list available [created by Rachel Tatman](https://www.kaggle.com/datasets/rtatman/english-word-frequency). A copy is hosted [here](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv).
- The most frequent word in the paper consists of alpha characters only, and it appears in less than 7.5% of the document.
  - Words are obtained by splitting the text on whitespace.


Train data is anything published before 2022-12-01; validation data is anything published after 2022-12-01 and until 2023-01-03.

|Split|Documents|Approx Word Count|Location|
|---|---|---|---|
train|9,933,204|43,363,567,649|`s3://ai2-s2-research-public/lucas/s2_oa_pretrain_data/v2/s2orc/train`
validation|119,902|640,156,445|`s3://ai2-s2-research-public/lucas/s2_oa_pretrain_data/v2/s2orc/valid`

## S2AG (Titles and Abstracts Papers)

The S2AG corpus contains titles and abstracts of papers in Semantic Scholar.
Unfiltered, the corpus contains 91.1M papers and 15.5B whitespace-separated tokens, but we impose the following constraints:

- Abstract must be in English.
  - To calculate the language, we once again use pycld3
- Title must be in English, or have average unigram log probability greater than -20.
- Abstract must be in English.
- Abstract must have higher than -20 average unigram log probability.
- Abstract must have at least 50 words.
- Abstract must have no more than 1000 words.
- The most frequent word in the union of text and abstract must be a 2+ character alpha word, or it can be `a` followed by a 2+ character alpha word.
- Paper was published after 1969.

|Split|Abstracts|Approx Word Count|Location|
|---|---|---|---|
train|59,161,786|10,894,621,062|`s3://ai2-s2-research-public/lucas/s2_oa_pretrain_data/v2/s2ag/train`
validation|118,357|25,828,000|`s3://ai2-s2-research-public/lucas/s2_oa_pretrain_data/v2/s2ag/valid`

## Format

Data is available in JSONL format at the following locations:

```
s3://ai2-s2-research-public/lucas/s2_oa_pretrain_data/
|-- v2/
|   |-- s2ag/
|   |   |-- train/
|   |   |-- valid/
|   |-- s2orc/
|   |   |-- train/
|   |   |-- valid/
```

Each directory contains 30 gzipped files, each of which contains a JSONL file. Each line contains the following keys:
- `id`: The paper ID.
- `sha1` (optional): The SHA1 hash of the paper.
- `text`: The text of the paper. Sections are separated by two newlines, i.e. `\n\n`; paragraphs are separated by a single newline, i.e. `\n`.
  - For full text papers, each text looks like `[title]\n\n[abstract]\n\n[section header]\n[paragraph]\n\n[paragraph]\n\n[section header]\n\n[paragraph]\n\n[...]`
  - For titles and abstracts, each text looks like `[title]\n\n[abstract]`
