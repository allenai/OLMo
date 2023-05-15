# S2 Dataset

## Tagged dataset (V4)

We tag with the following information:

Abstracts:

- `title`: Title of the paper
- `abstract`: Abstract of the paper
- `fstt_language_title`: Language of the title as detected by FastText
- `cld2_language_title`: Language of the title as detected by cld2
- `cld3_language_title`: Language of the title as detected by cld3
- `fstt_language_abstract`: Language of the abstract as detected by FastText
- `cld2_language_abstract`: Language of the abstract as detected by cld2
- `cld3_language_abstract`: Language of the abstract as detected by cld3
- `upp_perplexity_title`: Perplexity of the title as detected by using unigram language model derived from the 1T Web Ngram corpus.
- `ccnet_perplexity_title`: Perplexity of the title as detected by using kenLM model from the CCNet pipeline.
- `upp_perplexity_abstract`: Perplexity of the abstract as detected by using unigram language model derived from the 1T Web Ngram corpus.
- `ccnet_perplexity_abstract`: Perplexity of the abstract as detected by using kenLM model from the CCNet pipeline.
- `title_count`: Count of whitespace-separated tokens in the title.
- `abstract_count`: Count of whitespace-separated tokens in the abstract.
- `top_frequencies`: Top 10 most frequent words in the union of the title and abstract (words are again whitespace separated tokens).
- `year`: Year of publication of the abstract.


For the [1T Web Ngram corpus](https://catalog.ldc.upenn.edu/LDC2006T13), we specifically use the list available [created by Rachel Tatman](https://www.kaggle.com/datasets/rtatman/english-word-frequency). A copy is hosted [here](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv).


## Filtered version (V???)

TBD
