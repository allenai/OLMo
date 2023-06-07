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
conda install -c anaconda protobuf -y
python -m smashed.utils.install_blingfire_macos
```

If you are running on a bare EC2 instance, you will need to install `gcc` and `protobuf`, e.g. on Ubuntu:

```bash
sudo apt install build-essential protobuf-compiler -y
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
