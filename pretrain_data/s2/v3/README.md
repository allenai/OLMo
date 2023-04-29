# S2 Corpus v3

> *Author*: Luca Soldaini [@soldni](github.com/soldni)


Version 3 of the S2ORC corpus improves over 2 by removing abstracts from sources that
are not high-quality. For example, we remove abstracts that have originated exclusively
from the Microsoft Academic Graph, as they

We identified the following sources to be of lower quality:

If an abstract is exclusively from one of these sources, we remove it from the corpus.


## Dataset Statistics

| Dataset | Split | # Documents | # Words        |
|:-------:|:-----:|------------:|---------------:|
| s2ag    | train | 30,569,017  |  5,920,099,207 |
| s2ag    | valid |    109,709  |     24,029,459 |
| s2orc   | train |  8,242,162  | 36,088,195,908 |
| s2orc   | valid |     51,323  |    255,139,074 |
