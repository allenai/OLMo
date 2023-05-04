# C4 cleaning and filtering scripts 

## Set up 

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

## Run the cleaning script 

```bash
python c4-cleaning.py --input_file=<path to input file> --output_file=<path to output file>
```

## Documentation 

The original [C4 code](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/c4.py) has done the following for filtering and cleaning the CC text:

1. Filter documents by length
    - based on # characters: if one doc has more than 1.9e5 characters in the text, it drops the doc 
    - based on # paragraphs: the doc needs to have more than 3 paragraphs, and it should contain paragraphs of more than 200 characters
    - ... 
    - based on bad words 
2. Only keep documents from certain domain names 
    - there are some pre-specified lists of domain names including “news” like or web-text like 
3. Deduplicate contents 
    - based on their urls; if contents are in the same url, choose the latest one 
    - remove duplicate lines found across text documents (it indexes the text using a hash table, and then remove the same text in-place expect for one document) 
4. Clean page text 
    - remove citation from Wikipedia pages (among others)
    - some clean heuristics, which can be very easily copied and implemented 
    - line_has_too_long_word or too short 
    - remove docs which probably contain javascript code 
    - ... 

Current our code has most of the above functionalities, except for the following:

### TODOs

- [ ] Add filtering based on the links 
- [ ] Add deduplication based on the links (keeping only the newer file)
- [ ] Add deduplication based on text 
