import numpy as np
import pandas as pd
from collections import Counter
import math

class CustomTfidfVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.idf_scores = {}
        self.feature_names = []

    def fit_transform(self, corpus):
        tf_scores = [self.compute_tf(doc) for doc in corpus]
        self.compute_idf(corpus)

        # Constructing the TF-IDF matrix
        tf_idf_matrix = []
        for tf in tf_scores:
            doc_tf_idf = [tf.get(word, 0) * self.idf_scores.get(word, 0) for word in self.feature_names]
            tf_idf_matrix.append(doc_tf_idf)

        return np.array(tf_idf_matrix)

    def compute_tf(self, document):
        word_counts = Counter(document.lower().split())
        total_words = sum(word_counts.values())
        return {word: count / total_words for word, count in word_counts.items()}

    def compute_idf(self, corpus):
        doc_count = len(corpus)
        word_doc_count = Counter(word for document in corpus for word in set(document.lower().split()))
        self.idf_scores = {word: math.log(doc_count / (1 + freq)) for word, freq in word_doc_count.items()}
        sorted_words = sorted(self.idf_scores.items(), key=lambda item: item[1], reverse=True)
        self.feature_names = [word for word, idf in sorted_words[:self.max_features]]

    def get_feature_names_out(self):
        return self.feature_names

class TextProcessor:
    def __init__(self, df, max_features=1000):
        self.df = df
        self.tfidf_matrix = None
        self.vectorizer = CustomTfidfVectorizer(max_features=max_features)

    def apply_tfidf(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Text'])
        feature_names = self.vectorizer.get_feature_names_out()
        return pd.DataFrame(self.tfidf_matrix, columns=feature_names)
