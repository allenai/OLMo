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

## TODOs

- [ ] Add filtering based on the links 
- [ ] Add deduplication based on the links (keeping only the newer file)
- [ ] Add deduplication based on text 
