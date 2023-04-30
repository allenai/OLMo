import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class LRClassifier:
    def __init__(self):
        data_dir = "/net/nfs.cirrascale/mosaic/khyathic/projects/llm_data/jigsaw_data/toxic-comment"
        train_file = os.path.join(data_dir, "train.csv")
        # Load the dataset
        data = pd.read_csv(train_file)
        df = pd.DataFrame(data, columns=["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
        comment_texts = df['comment_text'].tolist()
        toxic_column_names = ["toxic", "severe_toxic", "threat", "insult", "identity_hate"]
        toxic_labels = df[toxic_column_names].sum(axis=1)
        toxic_labels = [1 if x >= 1 else 0 for x in toxic_labels.tolist()]

        # self.texts = ["I like you", "I hate you"]*20  # list of text samples
        # self.labels = [0, 1]*20  # list of corresponding labels (0 or 1)
        self.texts = comment_texts
        self.labels = toxic_labels

        # self.train_texts = ["I like you", "I hate you"]*20  # list of text samples
        # self.train_labels = [0, 1]*20  # list of corresponding labels (0 or 1)
        self.vectorizer = CountVectorizer()
        self.classifier = LogisticRegression()

    def train(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)

        # Convert the text to a bag-of-words representation
        X_train = self.vectorizer.fit_transform(X_train)

        X_test = self.vectorizer.transform(X_test)
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    def infer(self, X_test):
        X_test = self.vectorizer.transform(X_test)
        # y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        return y_pred_proba


'''
lr = LRClassifier()
lr.train()
lr.infer(None)
'''
