# Data Filters

General implementation of document-level scoring functions used for data filtering.

## Setup

To install the filters package, simply run:

```bash
pip install pretrain_data/filters
```

To install the package in development mode (e.g., if you are adding taggers), run:

```bash
pip install -e pretrain_data/filters'[dev]'
```

If you are planning to run this code on a Mac with Apple Silicon, you will need to run the following to install blingfire (used for sentence splitting):

```bash
python -m smashed.utils.install_blingfire_macos
```

## Running Taggers

To run taggers, use the following command:

```bash
ai2_llm_filters \
    -d "<name of dataset>" \
    -n "<experiment name>" \
    -t <tagger 1> ... <tagger n>  \
    -p <num processes>
```

To get a list of all options, run `ai2_llm_filters -l`.

## Adding a Tagger

Taggers should be added to `src/ai2_llm_filters/taggers`.
See `src/ai2_llm_filters/taggers/sampling.py` for an example.
Essentially, taggers should implement a `predict(self, doc: Document) -> DocResult` method that takes a document
and return a `DocResult` object.
For example, here's a tagger that returns a boolean if a text starts with the letter 'a':

```python
from ai2_llm_filters import BaseTagger, DocResult, Document, TaggerRegistry, Span

@TaggerRegistry.add("starts_with_a_document_v1")
class SampleTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        starts_with_a = doc.text.startswith("a")
        spans = [Span(start=0, end=len(doc.text), label="starts_with_a")] if starts_with_a else []
        return DocResult(doc=doc, spans=spans)
```

Your tagger could also return a score in the [0.0, 1.0] range:

```python
from ai2_llm_filters import BaseTagger, DocResult, Document, TaggerRegistry, Span

@TaggerRegistry.add("starts_with_a_document_v2")
class SampleTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        starts_with_a = doc.text.startswith("a")
        spans = [Span(start=0, end=len(doc.text), label="starts_with_a", score=1.0 if starts_with_a else 0.0)]
        return DocResult(doc=doc, spans=spans)
```

Multiple spans can be returned as well:

```python
from ai2_llm_filters import BaseTagger, DocResult, Document, TaggerRegistry, Span
from ai2_llm_filters.core_tools.utils import split_paragraphs

@TaggerRegistry.add("starts_with_a_paragraph_v1")
class SampleTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans = []
        for para in paragraphs:
            starts_with_a = para.text.startswith("a")
            score = 1.0 if starts_with_a else 0.0
            spans.append(Span(start=para.start, end=para.end, label="starts_with_a", score=score))
        return DocResult(doc=doc, spans=spans)
```

Note that, in case you create a new file under `src/ai2_llm_filters/taggers`, you need to import it in `src/ai2_llm_filters/taggers/__init__.py`.

### Creating a training dataset for a FastText Tagger

A binary classification dataset in the FastText format can be created using the `ft_dataset.py` module. The following example invocation will create a training file with 2 labels -- `__c4__` (positive) and `__gutenberg__` (negative) -- where a FastText tagger trained on the file can be trained to predict segments that are in C4 (using examples from Gutenberg as non-C4 examples). Each example has just one label. The flag `-t` is the dataset used for the target (positive) examples. The flag `-s` specifies one or more negative datasets to use.

```bash
python -m ai2_llm_filters.core_tools.ft_dataset \
  -t s3://ai2-llm/pretraining-data/sources/c4/v0/documents/train/ \
  -s s3://ai2-llm/pretraining-data/sources/gutenberg/v0/documents/ \
  -m sentence \
  --newlines skip \
  -o ./test-output.txt \
  --pos-label c4 \
  --neg-label gutenberg \
  --n-segments 3
```

The final training file will include at most 6 examples (3 sentences of both positive and negative). The file is NOT shuffled and should be shuffled on disk or during training if desired. Since files are processed in parallel from the target and then the source datasets there may be some random ordering within a class but all negative examples follow all positive.

Specifying `--n-segments` is HIGHLY recommended otherwise the entire dataset will be downloaded. If `--n-segments` is provided only enough dataset files are downloaded to keep the parallel worker processes busy. The file construction will terminate once enough examples have been generated.

## Output Format

Assuming you are running the pipeline for dataset `dataset_name/v_n` and have experiment name `exp_p`, the output will be stored in `s3://ai2-llm/pretraining-data/sources/dataset_name/v_n/attributes/exp_p`.

Assuming that you have run taggers `tagger_1` to `tagger_m`, each attribute file is a gzip'ed JSONL file with the following format:

```json
{
  "source": "dataset_name",
  "id": "<document id>",
  "attributes" : {
        "exp_p__tagger_1__type_1": [[<start_1>, <end_1>, <score_1>], ..., [<start_k>, <end_k>, <score_k>]],
        ...,
        "exp_p__tagger_1__type_i": [[<start_1>, <end_1>, <score_1>], ..., [<start_h>, <end_h>, <score_h>]]
        ...,
        "exp_p__tagger_m__type_1": [[<start_1>, <end_1>, <score_1>], ..., [<start_l>, <end_l>, <score_l>]]
        ...,
        "exp_p__tagger_m__type_j": [[<start_1>, <end_1>, <score_1>], ..., [<start_p>, <end_p>, <score_p>]]
    }
}
```

For taggers/types that apply to the entire document, the `start` and `end` values are set to `0` and `len(doc.text)`, respectively:

```json
{
  "source": "dataset_name",
  "id": "<document id>",
  "attributes" : {
        "exp_p__tagger_1__type_1": [[0, len(doc.text), <score>]],
        ...,
        "exp_p__tagger_m__type_j": [[0, len(doc.text), <score>]]
    }
}
```
