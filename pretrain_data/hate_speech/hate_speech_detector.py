import argparse
import json
import os
import timeit
import csv
import pickle

import fasttext
from nltk.tokenize.punkt import PunktSentenceTokenizer

from LRClassifier import LRClassifier

def predict(model_type, classifier, text):
    if model_type == "lr":
        pred_probs = classifier.infer([text])[0]
    if model_type == "fasttext":
        # k=-1 returns probabilities for all classes instead of just chosen class
        pred_label, pred_probs = classifier.predict(text, k=-1)
    return pred_probs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="JSON file containing text on which hate speech detection needs to be run")
    parser.add_argument("--model", type=str, required=True, help="Choose hate speech detector to run from [blocklist, svm, lr, toxicbert, roberta]")
    parser.add_argument("--process_doc_level", type=bool, default=False, help="Run at document Level")
    parser.add_argument("--metadata_file", type=str, default='metadata_annotations.json', required=True, help="Metadata JSON file")
    parser.add_argument("--load_pretrained", type=bool, default=False, required=False, help="Load a pretrained checkpoint if True, else train from scratch")
    parser.add_argument("--sent_threshold", type=float, help="Set probability threshold above which a sentence is considered hateful")
    args = parser.parse_args()

    print("Setting up tokenizers and classifiers...")
    sent_tokenizer = PunktSentenceTokenizer()
    classifier = None
    if args.model == "fasttext":
        if args.load_pretrained:
            classifier = fasttext.load_model("jigsaw_fasttext_bigrams.bin")
        else:
            classifier = fasttext.train_supervised(input="jigsaw_fasttext.train", epoch=100, wordNgrams=2)
            classifier.save_model("jigsaw_fasttext_bigrams.bin")
    elif args.model == 'lr':
        classifier = LRClassifier()
        if args.load_pretrained:
            classifier = pickle.load(open('jigsaw_trained_lr.pkl', 'rb'))
        else:
            classifier.train()

    start_time = timeit.default_timer()

    # Load data from input file
    reader = open(args.data_file, "r")
    attribute_outputs = []
    for line in reader:
        data = json.loads(line)
        if args.process_doc_level:
            # Perform prediction at document level
            # FastText complains about newlines so replace them with spaces
            text = data['text'].replace("\n", " ")
            pred_probs = predict(args.model, classifier, text)
            score = pred_probs[1]
            score_type = "fasttext_bigram_jigsaw_doc" if args.model == "fasttext" else "lr_jigsaw_doc"
            attribute_outputs.append(
                {
                    "id": data['id'],
                    "source": data['source'],
                    "hate_doc": (score, score_type),
                    # "text": data['text'] 
                }
            )
        else:
            # Perform prediction at sentence level
            sent_span_indices = sent_tokenizer.span_tokenize(data['text'])
            doc_attributes = {
                "id": data['id'],
                "source": data['source'],
                # "text": data['text'], 
                "hate": []
            }
            hate_sentence_count = 0.0
            total_sentence_count = 0.0
            for start,end in sent_span_indices:
                total_sentence_count += 1
                sentence_text = data['text'][start:end].replace("\n", " ")
                pred_probs = predict(args.model, classifier, sentence_text)
                score = pred_probs[1]
                score_type = "fasttext_bigram_jigsaw_sent" if args.model == "fasttext" else "lr_jigsaw_sent"
                doc_attributes["hate"].append([start, end, score, score_type])
                hate_sentence_count += 1 if score > args.sent_threshold else 0   # Can tweak this and experiment with different thresholds
                # if score > args.sent_threshold:
                #     print(sentence_text)
            doc_attributes["hate_doc"] = ((float(hate_sentence_count)/total_sentence_count), "hateful_sentence_ratio")
            attribute_outputs.append(doc_attributes)

    elapsed = timeit.default_timer() - start_time
    print("Processed {} documents in {:.6f} s".format(len(attribute_outputs), elapsed))

    writer = open(args.metadata_file, "w")
    for sample in attribute_outputs:
        writer.write(json.dumps(sample)+"\n")
    writer.close()

    # Sort documents by their hate speech scores and dumps them for precision analysis
    # k = 50
    # ranked_docs = sorted(attribute_outputs, key=lambda x: x["hate_doc"][0], reverse=True)
    # top_k_docs = ranked_docs[:k]
    # bottom_k_docs = ranked_docs[-k:]
    # middle_k_docs = ranked_docs[int(len(ranked_docs)/2)-int(k/2):int(len(ranked_docs)/2)+int(k/2)+1]

    # def dump_docs(docs, filename):
    #     out = open(os.path.join(filename), "w")
    #     csv_writer = csv.writer(out)
    #     csv_writer.writerow(["id", "source", "hate_speech_score", "text"])
    #     for sample in docs:
    #         csv_writer.writerow([sample["id"], sample["source"], sample["hate_doc"][0], sample["text"]])
    #     out.close()
    
    # dump_docs(top_k_docs, "top_docs.csv")
    # dump_docs(bottom_k_docs, "bottom_docs.csv")
    # dump_docs(middle_k_docs, "middle_docs.csv")

    # if args.model == "lr":  # 95.798
    #     print("Predicting with LR")
    #     all_preds_probs = []
    #     n_win = int(args.n_win)
        
    #     hate_speech_labels = []
    #     start_time = timeit.default_timer()
    #     for i, sentence in enumerate(sentences[: -n_win]):
    #         combined_sentences_list = sentences[i : i + n_win]
    #         combined_sentence = " ".join(combined_sentences_list)
    #         pred_proba = lr.infer([combined_sentence])[0]
    #         score = pred_proba[1]
    #         all_preds_probs.append(score)
    #         # hate_speech_labels.append(pred)
    #     elapsed = timeit.default_timer() - start_time
    #     print("Processed {} sentences in {:.6f} s".format(len(sentences), elapsed))
    #     print("{} out of {} sentences contain hate speech".format(sum(hate_speech_labels), len(sentences)))
    
    # if os.path.exists(args.metadata_file):
    #     with open(args.metadata_file) as f:
    #         metadata = json.load(f)
    #     metadata = {"hate_doc": all_preds_probs}
    #     with open(args.metadata_file, 'w') as f:
    #         json.dump(metadata, f)
        
    #     out = open(args.preds_file, 'w')
    #     for i, score in enumerate(all_preds_probs):
    #         if float(score) > 0.9:
    #             out.write(sentences[i]+'\n')
    #     out.close()