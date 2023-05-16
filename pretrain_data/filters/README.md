# Data Filters

General implementation of document-level scoring functions used for data filtering.

## Setup

```bash
pip install -e '.[dev]'
python -m spacy download en_core_web_lg
```


## Run Taggers

```bash
python -m ai2_llm_filters.taggers \
    dataset="<name of dataset>" \
    name="<name for group of taggers>" \
    taggers="['<tagger1>', '<tagger2>']"  \
    num_processes=<num processes>
```
