import argparse
from detoxify import Detoxify
import json
import LRClassifier
from nltk import sent_tokenize, word_tokenize
import timeit
import torch
# import nltk
# import pandas as pd


def read_blocklist(file):
    terms = open(file).read().splitlines()
    return set(terms)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="JSON file containing text on which hate speech detection needs to be run")
    parser.add_argument("--model", type=str, required=True, help="Choose hate speech detector to run from [blocklist, svm, lr, toxicbert, roberta]")
    parser.add_argument("--preds_file", type=str, required=True, help="Output file to dump detected hate speech for precision evaluation")
    parser.add_argument("--n_win", type=int, required=True, help="Window of the sentences")
    parser.add_argument("--process_doc_level", type=str, default='true', required=True, help="Document Level")
    parser.add_argument("--metadata_file", type=str, default='metadata_annotations.json', required=True, help="Metadata JSON file")
    args = parser.parse_args()

    # Load data from input file
    reader = open(args.data_file, "r")
    data = [json.loads(line) for line in reader]
    print("Evaluating {} texts for hate speech".format(len(data)))

    # Split texts into sentences
    # Can be replaced with any sentence splitter as needed
    # Timing tokenization separately for now
    sentences = []
    start_time = timeit.default_timer()
    for sample in data:
        text = sample['text']
        if args.process_doc_level == 'true':
            sentences.append(text)
            continue
        split_sentences = sent_tokenize(text)
        if args.model in ['blocklist', 'svm', 'lr']:
            sentences += [" ".join(word_tokenize(x)) for x in split_sentences]
        elif args.model in ['toxicbert', 'roberta']:
            sentences += split_sentences
    elapsed = timeit.default_timer() - start_time
    print("Tokenized {} sentences in {:.6f} s".format(len(sentences), elapsed))

    if args.model == "blocklist":
        # Use a blocklist to screen for hateful terms - likely to be fast, high precision, low recall
        # We are using this list: https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en 
        start_time = timeit.default_timer()
        hate_speech_labels = []
        blocked_terms = read_blocklist("en-blocklist.txt")
        for sentence in sentences:
            num_block_terms_present = len([x for x in blocked_terms \
                if " "+x+" " in sentence or \
                sentence.startswith(x) or \
                sentence.endswith(x)])
            # num_block_terms_present = len(set(sentence).intersection(blocked_terms))
            if num_block_terms_present > 0:
                hate_speech_labels.append(1)
            else:
                hate_speech_labels.append(0)
        elapsed = timeit.default_timer() - start_time
        print("Processed {} sentences in {:.6f} s".format(len(sentences), elapsed))
        print("{} out of {} sentences contain blocked terms".format(sum(hate_speech_labels), len(sentences)))
        out = open(args.preds_file, 'w')
        for i, label in enumerate(hate_speech_labels):
            if label == 1:
                out.write(sentences[i]+'\n')
                # out.write(' '.join(sentences[i])+'\n')
        out.close()

    # if args.model == "svm":
    if args.model == "lr": #95.798
        print ("Predicting with LR")
        all_preds_probs = []
        n_win = int(args.n_win)
        lr = LRClassifier()
        lr.train()
        hate_speech_labels = []
        start_time = timeit.default_timer()
        for i, sentence in enumerate(sentences[:-n_win]):
            combined_sentences_list = sentences[i:i+n_win]
            combined_sentence = " ".join(combined_sentences_list)
            pred_proba = lr.infer([combined_sentence])[0]
            score = pred_proba[1]
            all_preds_probs.append(score)
            #hate_speech_labels.append(pred)
        elapsed = timeit.default_timer() - start_time
        print("Processed {} sentences in {:.6f} s".format(len(sentences), elapsed))
        print("{} out of {} sentences contain hate speech".format(sum(hate_speech_labels), len(sentences)))
        if os.path.exists(args.metadata_file):
            with open(args.metadata_file) as f:
                metadata = json.load(f)
        metadata = {"hate_doc": all_preds_probs }
        with open(args.metadata_file, 'w') as f:
            json.dump(metadata, f) 
        '''
        out = open(args.preds_file, 'w')
        for i, label in enumerate(hate_speech_labels):
            if label == 1:
                out.write(sentences[i]+'\n')
        out.close()
        '''

    if args.model == "toxicbert":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print ("Predicting with ToxicBert")
        hate_speech_labels = []
        start_time = timeit.default_timer()
        for sentence in sentences:
            model = Detoxify('original', device=device)
            res = model.predict(sentence)
            toxic_score = res['toxicity']
            if toxic_score > 0.5:
                hate_speech_labels.append(1)
            else:
                hate_speech_labels.append(0)
        elapsed = timeit.default_timer() - start_time
        print("Processed {} sentences in {:.6f} s".format(len(sentences), elapsed))
        print("{} out of {} sentences contain hate speech".format(sum(hate_speech_labels), len(sentences)))
        out = open(args.preds_file, 'w')
        for i, label in enumerate(hate_speech_labels):
            if label == 1:
                out.write(sentences[i] + '\n')
        out.close()