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

`python run.py --infile ~/ai2/LLM/pretrain_data/c4/part_1017.jsonl.gz --outdir ~/ai2/LLM/pretrain_data/c4/ --method presidio --batch 100 --head 1000`

```
Elapsed time: 5.78 seconds
Elapsed time: 4.94 seconds
Elapsed time: 7.10 seconds
Elapsed time: 5.80 seconds
Elapsed time: 5.46 seconds
Elapsed time: 5.65 seconds
```

`python run.py --infile ~/ai2/LLM/pretrain_data/c4/part_1017.jsonl.gz --outdir ~/ai2/LLM/pretrain_data/c4/ --method presidio --postprocess --batch 100 --head 1000`

```
Elapsed time: 7.23 seconds
Elapsed time: 5.00 seconds
Elapsed time: 7.49 seconds
Elapsed time: 5.98 seconds
Elapsed time: 5.74 seconds
```

2. **Postprocessing doesnt affect Presidio much**. 

Only a single phone number detected by Presidio was removed by Postprocessing:
```
{"pii": [[464, 478, "PHONE_NUMBER", "1119077370 432"]], "score": 0.0018214936247723133}
```

3. **Regex so fast even w postprocess**

`python run.py --infile ~/ai2/LLM/pretrain_data/c4/part_1017.jsonl.gz --outdir ~/ai2/LLM/pretrain_data/c4/ --method regex --postprocess --batch 100 --head 1000`

```
Elapsed time: 0.02 seconds
Elapsed time: 0.02 seconds
Elapsed time: 0.03 seconds
Elapsed time: 0.03 seconds
Elapsed time: 0.03 seconds
```

4. **Regex with Postprocessing is approx same as Presidio**

See scratchwork at `compare.py`. Main takeaway is that they almost always agree on which docs. Presidio catches more things but manually inspecting, it's not even clear these are problematic PII that need to be avoided.

```
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
	Presidio=(0.008849557522123894, ['734-769-0001', '734-769-7677'])
\Regex=(0.008849557522123894, [' 734-769-0001', ' 734-769-7677'])
	Presidio=(0.02127659574468085, ['303-423-5119'])
\Regex=(0.02127659574468085, [' 303-423-5119'])
	Presidio=(0.001692047377326565, ['(614) 855-1103'])
\Regex=(0.001692047377326565, [' (614) 855-1103'])
	Presidio=(0.04225352112676056, ['sechrista@charlottesville.org', '434-970-3356', '(434) 989-6371'])
\Regex=(0.04225352112676056, [' sechrista@charlottesville.org.\n', ' 434-970-3356', ' (434) 989-6371'])
	Presidio=(0.009433962264150943, ['info@shunpike.org', '206 905 1026'])
\Regex=(0.0047169811320754715, [' 206 905 1026'])
	Presidio=(0.003663003663003663, ['jessica@blockchain.wtf'])
\Regex=(0.003663003663003663, [' jessica@blockchain.wtf '])
	Presidio=(0.01020408163265306, ['(580) 256-6465'])
\Regex=(0.01020408163265306, [' (580) 256-6465'])
	Presidio=(0.0019880715705765406, ['042 9370990'])
\Regex=(0.0019880715705765406, [' 042 9370990'])
	Presidio=(0.014084507042253521, ['312-361-0864'])
...
	Presidio=(0.007142857142857143, ['770-408-1001'])
\Regex=(0.007142857142857143, [' 770-408-1001'])
	Presidio=(0.008064516129032258, ['iusradio@ius.edu'])
\Regex=(0.008064516129032258, [' iusradio@ius.edu '])
```

#### Common Crawl

Ok, but that was using C4 which is already been filtered for PII. What about the (lowest perplexity) Common Crawl data?

`python run.py --infile ~/ai2/LLM/pretrain_data/common_crawl/cc_en_tail-0835.json.gz --outdir ~/ai2/LLM/pretrain_data/common_crawl/ --method presidio --postprocess --batch 100 --head 1000`

```
Elapsed time: 12.13 seconds
Elapsed time: 8.35 seconds
Elapsed time: 13.22 seconds
Elapsed time: 7.57 seconds
Elapsed time: 9.81 seconds
```

`python run.py --infile ~/ai2/LLM/pretrain_data/common_crawl/cc_en_tail-0835.json.gz --outdir ~/ai2/LLM/pretrain_data/common_crawl/ --method regex --postprocess  --batch 100 --head 1000`

```
Elapsed time: 0.05 seconds
Elapsed time: 0.04 seconds
Elapsed time: 0.06 seconds
Elapsed time: 0.03 seconds
Elapsed time: 0.04 seconds
```

The results from `compare.py` are:

```
Both agree no PII: 893 or 0.893

Presidio caught 54 that Regex didnt catch
	(0.0005861664712778429, ['168.00', '168.00'])
	(0.0002387448840381992, ['336-340', '334-343', '338-350', '11799-11800', '1698-1704', '( 0800 443 235', '( 0800 443 235'])
	(0.001524390243902439, ['::'])
	(0.04424778761061947, ['11-30-2012 02', '12-15-2012 12', '02-28-2012 10', '02-08-2012 10', '08-31-2011 04'])
	(0.056818181818181816, ['04-08-2012 02', '12-26-2019 10', '09-20-2020 11', '09-28-2020 12', '09-18-2020 05', '09-28-2020 12', '09-20-2020 01', '09-20-2020 12', '09-14-2020 12', '09-24-2020 10', '08-15-2016 09', '08-25-2020 10', '09-22-2020 07', '07-04-2006 09', '09-20-2020 07'])
	(0.0011918951132300357, ['08.02.2018 08.02.2018'])
	(0.0007451564828614009, ['1-800-832-2412'])
	(0.0017006802721088435, ['0234702'])
	(0.000502008032128514, ['0171-831 6394'])
	(0.01775147928994083, ['09958916872', '+91-9958916872', '09958916872'])
	(0.0012569581612497755, ['561-568. 2007', '0733-9429', '159-172. 2006', '519-539. 2005', '390-398', '0733-9429', '12(1784). 2008'])
	(0.01694915254237288, ['01926 811519'])
	(0.07142857142857142, ['webmaster@grow.co.uk'])
	(0.0003675119441381845, ['+86-13567774222'])
	(0.00011849745230477545, ['0 6 160 110 3'])
	(0.00017142367360932544, ['+923153528777', '+923153528777'])
	(0.0018281535648994515, ['170 260 67 67'])
	(0.0011363636363636363, ['+91 22 43431313'])
	(0.0017921146953405018, ['+86-13880247006', '0086-21-61182423'])
	(0.01639344262295082, ['08 2013 02'])
	(0.005847953216374269, ['halaabdullahi@hotmail.com'])
	(0.0010330578512396695, ['+1.781.392.2000'])
...
	(0.015852047556142668, ['2019.09.05 23', '2019.09.06 01', '2019.09.06 14', '2020.08.17 32', '2020.08.16 53', '2020.08.13 69', '2020.08.12 60', '2020.08.10 79', '2020.08.05 98', '2020.08.05 74', '2020.08.03 97', '2020.07.25 91'])
	(0.02734375, ['2018.08.01 11', '2018.08.01 12', '2018.08.01 21', '2018.08.02 02', '2018.10.19 76', '2018.10.18 95', '2018.10.18 93'])
	(0.007957559681697613, ['02 6113 0549', '02 6113 0549', '02 6113 0549'])

Regex caught 4 that Presidio didnt catch
	(0.001445086705202312, ['3.0.2.3'])
	(0.002824858757062147, ['\n9588588977'])
	(0.000308546744831842, [' 9781441303'])
	(0.029411764705882353, [' (1822011163'])

Docs both caught have PII: 49 or 0.049
	Presidio=(0.0011672016340822876, ['219 861-8740', '317 802-9686', '304-598-4848', '::'])
\Regex=(0.0008754012255617158, [' 219 861-8740', ' 317 802-9686', ' 304-598-4848'])
	Presidio=(0.005420054200542005, ['(019755 62253', '(019755) 62253'])
\Regex=(0.0027100271002710027, [' (019755 6225'])
	Presidio=(0.038461538461538464, ['979-596-2479', '979-596-2479', '979-596-2479', '979-596-2479', '979-229-1103', '(979) 885-3554', '979-224-4698', '979-446-2054', '979-218-8336', '936-539-3324'])
\Regex=(0.038461538461538464, [' 979-596-2479', ' 979-596-2479', ' 979-596-2479', ' 979-596-2479', ' 979-229-1103', ' (979) 885-3554', ' 979-224-4698', ' 979-446-2054', ' 979-218-8336', ' 936-539-3324'])
	Presidio=(0.010752688172043012, ['(800) 288-0014'])
\Regex=(0.010752688172043012, [' (800) 288-0014'])
	Presidio=(0.011111111111111112, ['(602) 273-6770'])
\Regex=(0.011111111111111112, [' (602) 273-6770'])
	Presidio=(0.004758128469468675, ['612-310-4832', '0466713111', '702-313-3300', '1226539429', '208-407-8889', '901-425-0970'])
\Regex=(0.0055511498810467885, [' 612-310-4832', ' 0466713111', ' 702-313-3300', ' 1226539429', ' 925-337-9045', ' 208-407-8889', ' 901-425-0970'])
	Presidio=(0.15384615384615385, ['35.37.39.41', '35.37.39.41.43'])
\Regex=(0.07692307692307693, ['35.37.39.41'])
	Presidio=(0.0025906735751295338, ['vivian@cnjnkj.com'])
\Regex=(0.0025906735751295338, [': vivian@cnjnkj.com\n'])
	Presidio=(0.0057306590257879654, ['enterplast@enterplast.com', 'enterplast@enterplast.com'])
\Regex=(0.0057306590257879654, [': enterplast@enterplast.com\n', 'enterplast@enterplast.com\n'])
	Presidio=(0.02564102564102564, ['steve.vangrouw@newhorizonstrans.com', '570.704.8860'])
\Regex=(0.02564102564102564, ['\tsteve.vangrouw@newhorizonstrans.com ', '\t570.704.8860'])
	Presidio=(0.0041841004184100415, ['chassaing.xavier@gmail.com'])
\Regex=(0.0041841004184100415, ['.\nchassaing.xavier@gmail.com\n'])
	Presidio=(0.003518029903254178, ['0429112', '85214-31010', '33542-20031', '8523748223'])
\Regex=(0.0008795074758135445, [' 8523748223'])
...
	Presidio=(0.002325581395348837, ['813 974-2171'])
\Regex=(0.002325581395348837, [' 813 974-2171'])
	Presidio=(0.009615384615384616, ['(870) 548-2291'])
\Regex=(0.009615384615384616, [' (870) 548-2291'])
```

This does look like Presidio is noticeably a bit better than Regexes as it's ability to find international phone numbers, though it also has a lot of false positives (some numbers that look like floating points values, dates, etc.).

**What about document-level ranking?**

What if we didn't worry about the individual PII entities, but instead just ranked the documents based on the number of PII entities found in them?  This would be a good way to find documents that are likely to have PII in them, and then we could use the individual entity scores to find the specific PII in the document.

First, let's look at the top 1% of scored docs. This would make sense if, say, we took a policy where we wanted to filter out the 1% of the data with the highest PII risk. What would those documents look like?

From the low perplexity Common Crawl set earlier, actually the regex method looks pretty good. It finds documents with email addresess, phone numbers, IP addreses. Presidio is a bit weird.:

```
Top 1% of docs is n=10 docs:
	Both agree high PII score: 5 or 0.5
		(0.07692307692307693, ['info@qk0.jxqqxjw.cn\n'])
		(0.05555555555555555, ['1.11.1.29', '1.11.1.29'])
		(0.0625, [' 902-701.4131'])
		(0.047619047619047616, [' 407-251-0669', ' 407-272-1417'])
		(0.07692307692307693, ['35.37.39.41'])
	Regex only: 5 or 0.5
		(0.029411764705882353, ['\nrazmyshkina@mail.ru\n'])
		(0.038461538461538464, ['\n20191103212204.13606-1-colin.king@canonical.com\n', ' <colin.king@canonical.com>)\n', ' <colin.king@canonical.com>\n', ' <alokc@codeaurora.org>,\n', ' <agross@kernel.org>, ', 'linux-i2c@vger.kernel.org,\n', ': kernel-janitors@vger.kernel.org, ', 'linux-kernel@vger.kernel.org\n', ': <20191103212204.13606-1-colin.king@canonical.com>\n', ' <colin.king@canonical.com>\n', ' <colin.king@canonical.com>\n', ' <colin.king@canonical.com>\n', ' <akashast@codeauror.org>\n', ' <colin.king@canonical.com>\n', ' <colin.king@canonical.com>\n', ' <akashast@codeauror.org>\n', ' <akashast@codeaurora.org>\n', '\n2019110321', '209.132.180.67', '91.189.89.112', '82.43.126.140'])
		(0.043478260869565216, ['3.235.101.50'])
		(0.029411764705882353, [' (1822011163'])
		(0.038461538461538464, [' 979-596-2479', ' 979-596-2479', ' 979-596-2479', ' 979-596-2479', ' 979-229-1103', ' (979) 885-3554', ' 979-224-4698', ' 979-446-2054', ' 979-218-8336', ' 936-539-3324'])
	Presidio only: 5 or 0.5
		(0.05714285714285714, ['zetataualpha@zetataualpha.org', '(317) 872-0540', '02-09-2015 3', '08-23-2018 3'])
		(0.07142857142857142, ['webmaster@grow.co.uk'])
		(0.04878048780487805, ['06-11-05 1024', '06-11-05 1024'])
		(0.056818181818181816, ['04-08-2012 02', '12-26-2019 10', '09-20-2020 11', '09-28-2020 12', '09-18-2020 05', '09-28-2020 12', '09-20-2020 01', '09-20-2020 12', '09-14-2020 12', '09-24-2020 10', '08-15-2016 09', '08-25-2020 10', '09-22-2020 07', '07-04-2006 09', '09-20-2020 07'])
		(0.2727272727272727, ['::', '::', '::'])
```

What if we removed the top 5% of documents with most PII?

```
Top 5% of docs is n=50 docs:
	Both agree high PII score: 23 or 0.46
		(0.009933774834437087, ['\n376.1956171', '\n188097.8085', '\n376195.6171'])
		(0.038461538461538464, [' 979-596-2479', ' 979-596-2479', ' 979-596-2479', ' 979-596-2479', ' 979-229-1103', ' (979) 885-3554', ' 979-224-4698', ' 979-446-2054', ' 979-218-8336', ' 936-539-3324'])
		(0.024390243902439025, [' sales@mrsafety.com '])
		(0.010752688172043012, [' (800) 288-0014'])
		(0.011111111111111112, [' (602) 273-6770'])
	Regex only: 27 or 0.54
		(0.002243829468960359, [' moobooks@verizon.net. ', ' genii@geniimagazine.com ', ' 301-652-5800'])
		(0.0008952551477170994, [' 800-210-7163'])
		(0.0008754012255617158, [' 219 861-8740', ' 317 802-9686', ' 304-598-4848'])
		(0.0024067388688327317, [': bornx2000@hotmail.com\n', ': bornx2000@yahoo.com\n'])
		(0.0025906735751295338, [': vivian@cnjnkj.com\n'])
	Presidio only: 27 or 0.54
		(0.04411764705882353, ['06-05-2006 06', '06-05-2006 12', '06-05-2006 11', '06-05-2006 06', '06-05-2006 07', '06-05-2006 02', '06-06-2006 12', '06-07-2006 01', '06-05-2006 09'])
		(0.009900990099009901, ['publications@epa.ie', '::'])
		(0.021052631578947368, ['02013026', '02013026'])
		(0.01775147928994083, ['09958916872', '+91-9958916872', '09958916872'])
		(0.012307692307692308, ['(2018.02.03-04', '2018.02.03-04', '(2017.02.18-19', '2017.02.18-19'])
```

Like with top 1% of docs, the two methods still agree about half the time. The remaining set... I still think Regex method is better, finding emails and phone numbers whereas Presidio is finding dates (though it's better at international phone numbers).

