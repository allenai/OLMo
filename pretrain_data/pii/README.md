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

## Benchmarking

#### C4

1. **Presidio is so slow...**

`python run.py --infile ~/ai2/LLM/pretrain_data/c4/part_1017.jsonl.gz --method presidio --outdir ~/ai2/LLM/pretrain_data/c4/ --batch 100 --head 1000`

```
Elapsed time: 4.85 seconds
Elapsed time: 7.19 seconds
Elapsed time: 5.61 seconds
Elapsed time: 5.37 seconds
Elapsed time: 5.43 seconds
Elapsed time: 3.97 seconds
```

`python run.py --infile ~/ai2/LLM/pretrain_data/c4/part_1017.jsonl.gz --method presidio --postprocess --outdir ~/ai2/LLM/pretrain_data/c4/ --batch 100 --head 1000`

```
Elapsed time: 5.52 seconds
Elapsed time: 4.90 seconds
Elapsed time: 7.06 seconds
Elapsed time: 5.66 seconds
Elapsed time: 5.56 seconds
Elapsed time: 5.52 seconds
```

2. **Postprocessing doesnt affect Presidio much**. 

Only a single phone number detected by Presidio was removed by Postprocessing:
```
{"pii": [[464, 478, "PHONE_NUMBER", "1119077370 432"]], "score": 0.0018214936247723133}
```

3. **Regex so fast even w postprocess**

`python run.py --infile ~/ai2/LLM/pretrain_data/c4/part_1017.jsonl.gz --method regex --postprocess --outdir ~/ai2/LLM/pretrain_data/c4/ --batch 100 --head 1000`

```
Elapsed time: 0.02 seconds
Elapsed time: 0.02 seconds
Elapsed time: 0.03 seconds
Elapsed time: 0.03 seconds
Elapsed time: 0.03 seconds
Elapsed time: 0.03 seconds
```

4. **Regex with Postprocessing is approx same as Presidio**

See scratchwork at `compare.py`. Main takeaway is that they almost always agree on which docs. Presidio catches more things but manually inspecting, it's not even clear these are problematic PII that need to be avoided.

```
Regex rows: 1000
Presidio rows: 1000
Both agree no PII: 948 or 0.948
Presidio caught 14 that Regex didnt catch
	(0.0011235955056179776, ['m.dunbabin@qut.edu.au'])
	(0.0008748906386701663, ['editor@cruisearabiaonline.com'])
	(0.011363636363636364, ['(01242) 251395'])
	(0.01098901098901099, ['0412-919-777'])
	(0.003003003003003003, ['1-800-396-1911'])
	(0.004201680672268907, ['1697'])
	(0.0024449877750611247, ['16.8 42.6'])
	(0.0022123893805309734, ['marketing@abbeytax.co.uk', '0345 223 2727'])
	(0.0038910505836575876, ['0401 993 880'])
	(0.004149377593360996, ['169–176'])
	(0.00909090909090909, ['0178 460 8100'])
	(0.005547850208044383, ['08.11.2013', '08.11.2013', '08.11.2013', '08.11.2013'])
	(0.012658227848101266, ['events@gla.org'])
	(0.0014245014245014246, ['04 – 04 1/2'])
Regex caught 1 that Presidio didnt catch
	(0.00211864406779661, [' 0000190000'])
Docs both caught have PII: 37 or 0.037
	Presidio=(0.0055248618784530384, ['(850) 654-9983'])
\Regex=(0.0055248618784530384, [' (850) 654-9983'])
	Presidio=(0.0026666666666666666, ['(662) 234-4336'])
\Regex=(0.0026666666666666666, [' (662) 234-4336'])
...
	Presidio=(0.007142857142857143, ['770-408-1001'])
\Regex=(0.007142857142857143, [' 770-408-1001'])
	Presidio=(0.008064516129032258, ['iusradio@ius.edu'])
\Regex=(0.008064516129032258, [' iusradio@ius.edu '])
```

