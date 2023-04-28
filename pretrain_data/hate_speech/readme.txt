This folder identifies the toxicity of the data.

The following 2 models are doing comparativelye better with a tradeoff of time and performance.
(1) Logistic Regression
(2) FastText

How to run?
The code takes as input a json chunk and dumps a json file with metadata annotations for the toxicity score.

python hate_speech_detector.py 
--model  <lr, fastext>
--data_file <path to olmo-data>/en_head.json 
--sent_threshold 0.7
--process_doc_level True
--metadata_file metadata_annotations.json


If metadata_annotations.json already exists, the script is expected to add a key corresponding to annotations among (pii, hate_speech, nsfw) etc., 
This script adds the key for 'hate_doc' in jsonl format.
Each line in the output jsonl  is "id": data['id'], "source": data['source'] (which are copied from the input file)
The hate speech is annotated with "hate_doc": (score, score_type).

Based on a threshold, the sentences can be identified later.
The precision is higher for threshold of 0.7 from our study for LR.

Notes: 
Time comparison: FastText better than LR.
Precision Score: LR > FastText.
Data identified as toxic text is higher for LR.


Other models that are explored include the following:

(1) Block Lists
(2) Logistic Regression
(3) SVM
(4) ToxicBERT
(5) FastText


