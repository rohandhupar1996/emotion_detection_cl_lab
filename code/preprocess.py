import pandas as pd
import numpy as np
import re
from collections import Counter
import math
import numpy as np


class SimpleCountVectorizer:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary_ = {}

    def fit(self, documents):
        word_counts = {}
        for doc in documents:
            tokens = doc.split()  # Tokenizing by spaces
            for token in tokens:
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
        
        # Sort by word frequency and limit the vocabulary to max_features most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]

        self.vocabulary_ = {word: index for index, (word, _) in enumerate(sorted_words)}

    def transform(self, documents):
        # Initialize matrix: Number of documents x Number of words in vocabulary
        X = np.zeros((len(documents), len(self.vocabulary_)))

        for idx, doc in enumerate(documents):
            tokens = doc.split()
            for token in tokens:
                if token in self.vocabulary_:
                    X[idx, self.vocabulary_[token]] += 1

        return X

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits the data into train and test sets.
    
    Parameters:
        X (numpy.ndarray): Feature dataset.
        y (numpy.ndarray): Labels corresponding to the feature dataset.
        test_size (float): The proportion of the dataset to include in the test split (between 0 and 1).
        random_state (int): A seed value for random number generation. Ensures reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test: arrays containing the split data.
    """
    if random_state:
        np.random.seed(random_state)
    
    # Total number of data points
    total_samples = X.shape[0]
    
    # Shuffle the indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Split indices for the training and test data
    test_size = int(total_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Slice the data to create training and testing subsets
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test



