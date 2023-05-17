# Data Filters

General implementation of document-level scoring functions used for data filtering.

## Setup

```bash
pip install -e '.[dev]'
python -m spacy download en_core_web_lg
```


## Running Taggers

To run taggers, use the following command:

```bash
python -m ai2_llm_filters \
    -d "<name of dataset>" \
    -n "<name for group of taggers>" \
    -t "<tagger 1>" -t "<tagger 2>"  \
    num_processes=<num processes>
```

To get a list of all options, run `python -m ai2_llm_filters.taggers -o`.


## Adding a Tagger

Taggers should be added to `src/ai2_llm_filters/taggers`.
See `src/ai2_llm_filters/taggers/sampling.py` for an example.
Essentially, taggers should implement a `tag(row: dict) -> dict` method that takes a row of data and returns a dictionary of tags.
For example, here's a tagger that returns a boolean if a text starts with the letter 'a':

```python
from .base import BaseTagger, TaggerRegistry

@TaggerRegistry.add("starts_with_a")
class SampleTagger(BaseTagger):

    def tag(self, row: dict) -> dict:
        return {"starts_with_a": row['text'].startswith('a')}
```

after adding a tagger, make sure to add an import statement to `src/ai2_llm_filters/taggers/__init__.py` so that it can be found by the registry.
