## PII Detection

This repository contains code to extract PII from text documents.

The types of PII detection currently supported are: 
+ Email Addresses, 
+ Phone Numbers, and 
+ IP Addresses.

The types of extractors currently supported are: 
+ Regular-expression based extractors, and 
+ [Presidio](https://github.com/microsoft/presidio). 

## Install

```
pip install presidio_analyzer
python -m spacy download en_core_web_lg
```

## How to run

```
python run_cc_shard_pp.py --in_file [OLMO SHARD] --classifier [regex/presidio] --output_file [OUTPUT_FILE]

```

## Outputs

We compute two additional fields for each document:
+ "pii": [(start<int>, end<int>, type<str>, match<str>), (start, end, type, match), ....]  where the start, end are integer-valued character-level indices. The type corresponds to the PII type ("email"/"phone_numbers"/"IP_addresses"). The match is the extracted string. Note that pii_start might correspond to -1 if the tokenization has changed.
+ "pii_doc": (score<float>) corresponds to the number of PII instances found in a document, normalized by the length of the document.
