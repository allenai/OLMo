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

#### Common Crawl

`python run.py --infile ~/ai2/LLM/pretrain_data/common_crawl/cc_en_head-1334.json.gz --method presidio --postprocess --outdir ~/ai2/LLM/pretrain_data/common_crawl/ --batch 100 --head 1000`

```
Elapsed time: 10.68 seconds
Elapsed time: 8.89 seconds
Elapsed time: 11.39 seconds
Elapsed time: 13.12 seconds
Elapsed time: 20.97 seconds
```

`python run.py --infile ~/ai2/LLM/pretrain_data/common_crawl/cc_en_head-1334.json.gz --method regex --postprocess --outdir ~/ai2/LLM/pretrain_data/common_crawl/ --batch 100 --head 1000`

```
Elapsed time: 0.05 seconds
Elapsed time: 0.04 seconds
Elapsed time: 0.05 seconds
Elapsed time: 0.06 seconds
Elapsed time: 0.10 seconds
```

**Comparison**

```
Regex rows: 1000
Presidio rows: 1000
Both agree no PII: 934 or 0.934
Presidio caught 22 that Regex didnt catch
	(0.3333333333333333, ['0406102'])
	(0.008264462809917356, ['1-888-767-7740'])
	(0.004080244814688882, ['2018.09.11 05', '2015.09.05 15', '2020.06.23 21', '2018.04.10 23', '2017.11.14 20', '2015.03.17 20', '2016.05.11 22', '2019.12.22 20', '2014.12.12 02', '2018.07.27 17', '2016.05.06 15', '2015.03.17 20'])
	(0.0044444444444444444, ['1127977966'])
	(0.0036900369003690036, ['1−800−799−7233'])
	(0.0027472527472527475, ['investor.relations@gulfnav.com'])
	(0.0008216926869350862, ['1140160276', '1250774349'])
	(0.009259259259259259, ['16889', '168898812'])
	(0.000741839762611276, ['meetings@unctad.org'])
	(0.0013280212483399733, ['1649908172'])
	(0.0021436227224008574, ['06.12.2020', '06.12.2020'])
	(0.000722543352601156, ['9303828733', '9303828733'])
	(0.00017051752067524938, ['93-0816-2371', '93-0920-2548'])
	(0.00037376191366099795, ['1648-1908', '1683-1932'])
	(0.0029940119760479044, ['press@xsb.com'])
	(0.000970873786407767, ['1683'])
	(0.002320185614849188, ['infotspower@gmail.com'])
	(0.006896551724137931, ['175 ( 2015 ) 314', '( 2015 ) 314 – 323'])
	(0.006944444444444444, ['albert@bg.legal'])
	(0.0020161290322580645, ['800‑256‑5169'])
	(0.004761904761904762, ['0.681818'])
...
	Presidio=(0.004827586206896552, ['communications@mnlct.org', 'info@mnlct.org', 'info@mnlct.org', '647-776-2057', '416-291-3248', '647-776-2057', '416-291-3248'])
\Regex=(0.004827586206896552, [': communications@mnlct.org.\n', ': info@mnlct.org.\n', ': info@mnlct.org.\n', ' 647-776-2057', ' 416-291-3248', ' 647-776-2057', ' 416-291-3248'])
	Presidio=(0.00033647375504710633, ['support@ecanarys.com'])
\Regex=(0.00033647375504710633, [' support@ecanarys.com '])

Regex caught 2 that Presidio didnt catch
	(0.04, [' 9781118391'])
	(0.0018726591760299626, ['\n2006200920'])

Docs both caught have PII: 42 or 0.042
	Presidio=(0.0007267441860465116, ['dpo@wphdigital.com'])
\Regex=(0.0007267441860465116, [' : dpo@wphdigital.com\n'])
	Presidio=(0.0033277870216306157, ['703-228-4241', '800-673-2777'])
\Regex=(0.0033277870216306157, [' 703-228-4241', ' 800-673-2777'])
	Presidio=(0.03333333333333333, ['(715) 341-8902'])
\Regex=(0.03333333333333333, [' (715) 341-8902'])
	Presidio=(0.002902757619738752, ['l.chappell@reading.ac.uk', '0751 518 8751'])
\Regex=(0.001451378809869376, [' l.chappell@reading.ac.uk\n'])
	Presidio=(0.0021598272138228943, ['(216) 432-0540'])
\Regex=(0.0021598272138228943, [' (216) 432-0540'])
	Presidio=(0.0016638935108153079, ['personnel@laurens55.org'])
\Regex=(0.0016638935108153079, ['\npersonnel@laurens55.org\n'])
	Presidio=(0.011204481792717087, ['adk@manleydeas.com', '2022 12 2145', '2022 12 2145', '614-220-5611'])
\Regex=(0.0056022408963585435, [' adk@manleydeas.com ', ' 614-220-5611'])
	Presidio=(0.0030911901081916537, ['info@xela.art', 'info@xela.art'])
\Regex=(0.0030911901081916537, [' info@xela.art. ', ' info@xela.art '])
	Presidio=(0.001968503937007874, ['954.831.4000'])
\Regex=(0.001968503937007874, [' 954.831.4000'])
	Presidio=(0.007518796992481203, ['recruitment@wessexwater.co.uk'])
\Regex=(0.007518796992481203, [': recruitment@wessexwater.co.uk\n'])
	Presidio=(0.002421307506053269, ['(876)792-1091', '(876)564-6690'])
\Regex=(0.002421307506053269, [' (876)792-1091', ' (876)564-6690'])
	Presidio=(0.0103359173126615, ['bradj@uoguelph.ca', 'emily.tolomei@tlraction.com', 'christopher.schiafone@tlraction.com', 'ldafoe@uoguelph.ca'])
\Regex=(0.0103359173126615, [' (bradj@uoguelph.ca) ', ' (emily.tolomei@tlraction.com) ', ' (christopher.schiafone@tlraction.com) ', ' (ldafoe@uoguelph.ca) '])
...
	Presidio=(0.004827586206896552, ['communications@mnlct.org', 'info@mnlct.org', 'info@mnlct.org', '647-776-2057', '416-291-3248', '647-776-2057', '416-291-3248'])
\Regex=(0.004827586206896552, [': communications@mnlct.org.\n', ': info@mnlct.org.\n', ': info@mnlct.org.\n', ' 647-776-2057', ' 416-291-3248', ' 647-776-2057', ' 416-291-3248'])
	Presidio=(0.00033647375504710633, ['support@ecanarys.com'])
\Regex=(0.00033647375504710633, [' support@ecanarys.com '])
```