''' 
Project: Multinomial Naive Bayes Classifier for Emotion Detections
Version: 0.1
Author: Rohan Dhupar

Description:
This module includes a custom implementation of a simple count vectorizer and a function to split data
into training and testing sets. These utilities are intended to preprocess text data for emotion detection
and to facilitate the evaluation of machine learning models.
'''

import pandas as pd
import numpy as np
import re
from collections import Counter
import math

class SimpleCountVectorizer:
    '''
    A simple implementation of a count vectorizer that converts a collection of text documents into
    a matrix of token counts, optionally limiting the number of features based on the most frequent tokens.
    
    Attributes:
        max_features (int, optional): The maximum number of most frequent features to include. If None, all features are included.
        vocabulary_ (dict): A mapping from feature names (tokens) to feature indices.
    '''

    def __init__(self, max_features=None):
        '''
        Initializes the SimpleCountVectorizer with an optional maximum number of features.
        
        Parameters:
            max_features (int, optional): The maximum number of features to consider based on term frequency.
        '''
        self.max_features = max_features
        self.vocabulary_ = {}

    def preprocess_text(self, text):
        '''
        Preprocesses the input text by converting to lowercase, removing URLs, mentions, hashtags, numbers,
        punctuation, and normalizing whitespaces.
        
        Parameters:
            text (str): The input text to preprocess.
        
        Returns:
            str: The preprocessed text.
        '''
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fit(self, documents):
        '''
        Learns the vocabulary dictionary of all tokens in the raw documents.
        
        Parameters:
            documents (list of str): A list of documents to learn the vocabulary from.
        '''
        word_counts = {}
        for doc in documents:
            doc = self.preprocess_text(doc)
            tokens = doc.split()
            for token in tokens:
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]

        self.vocabulary_ = {word: index for index, (word, _) in enumerate(sorted_words)}

    def transform(self, documents):
        '''
        Transforms the documents to a matrix of token counts using the learned vocabulary.
        
        Parameters:
            documents (list of str): The documents to transform.
        
        Returns:
            numpy.ndarray: The document-word matrix.
        '''
        X = np.zeros((len(documents), len(self.vocabulary_)))
        for idx, doc in enumerate(documents):
            doc = self.preprocess_text(doc)
            tokens = doc.split()
            for token in tokens:
                if token in self.vocabulary_:
                    X[idx, self.vocabulary_[token]] += 1

        return X

    def fit_transform(self, documents):
        '''
        Fits the vectorizer on the documents and then transforms them.
        
        Parameters:
            documents (list of str): The documents to fit and transform.
        
        Returns:
            numpy.ndarray: The document-word matrix.
        '''
        self.fit(documents)
        return self.transform(documents)


def train_test_split(X, y, test_size=0.2, random_state=None):
    '''
    Splits the data into train and test sets based on the test_size proportion.
    
    Parameters:
        X (numpy.ndarray): The features dataset.
        y (numpy.ndarray): The labels corresponding to the features.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int, optional): A seed value to ensure reproducibility of the shuffle.
    
    Returns:
        tuple: A tuple containing split data (X_train, X_test, y_train, y_test).
    '''
    if random_state:
        np.random.seed(random_state)

    total_samples = X.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    test_size = int(total_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
