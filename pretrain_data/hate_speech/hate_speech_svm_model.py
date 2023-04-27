from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import pandas as pd


class SVMClassifier:
    def __init__(self):
        data_dir = "/net/nfs.cirrascale/mosaic/khyathic/llm_data/jigsaw_data/toxic-comment"
        train_file = os.path.join(data_dir, "train.csv")
        # Load the dataset
        data = pd.read_csv(train_file)
        df = pd.DataFrame(data, columns=["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
        # comment_texts = df['comment_text'].tolist()
        # labels = df['toxic'].tolist()

        # self.train_texts = ["I like you", "I hate you"]*20  # list of text samples
        # self.train_labels = [0, 1]*20  # list of corresponding labels (0 or 1)
        self.vectorizer = CountVectorizer()
        self.classifier = svm.SVC()

    def train(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.train_texts, self.train_labels, test_size=0.2, random_state=42)

        # Convert the text to a bag-of-words representation
        X_train = self.vectorizer.fit_transform(X_train)

        X_test = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
