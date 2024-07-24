''' 
Project: Multinomial Naive Bayes Classifier for Emotion Detections
Version: 0.1
Author: Rohan Dhupar

Description:
This script implements a basic version of the Multinomial Naive Bayes classifier for emotion detection from text data.
The classifier calculates probabilities of each class based on the frequency of features and uses these probabilities
to make predictions.
'''

import numpy as np

class MultinomialNB:
    '''
    A Naive Bayes classifier for multinomial models which is particularly suited for feature vectors where the features
    represent the frequencies or counts of events/happenings.
    
    Attributes:
        class_log_prior_ (np.ndarray): Log probability of each class (using the prior probability).
        feature_log_prob_ (np.ndarray): Log probability of feature given a class.
        classes_ (np.ndarray): Unique classes in the target variable.
    '''

    def __init__(self):
        '''
        Initializes the MultinomialNB classifier with default values for class log priors, feature log probabilities,
        and class labels.
        '''
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
    
    def fit(self, X, y):
        '''
        Fit the Naive Bayes classifier according to X, a frequency matrix of features, and y, the target labels.
        
        Parameters:
            X (np.ndarray): The training input samples. Typically, X is a matrix of shape (n_samples, n_features),
                            where each element is the frequency of a feature in a sample.
            y (np.ndarray): The target values. An array of shape (n_samples,).
        '''
        # Count the number of samples in each class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.classes_ = unique_classes
    
        # Calculate log prior probabilities for each class
        total_samples = y.shape[0]
        self.class_log_prior_ = np.log(class_counts / total_samples)
        
        # Calculate the log probability of features given a class
        feature_counts = np.zeros((len(unique_classes), X.shape[1]))
        for idx, cls in enumerate(unique_classes):
            feature_counts[idx, :] = X[y == cls].sum(axis=0)
        
        # Smoothing
        smoothed_feature_counts = feature_counts + 1
        smoothed_class_sums = smoothed_feature_counts.sum(axis=1).reshape(-1, 1)
        
        self.feature_log_prob_ = np.log(smoothed_feature_counts / smoothed_class_sums)
        
    def predict(self, X):
        '''
        Perform classification on an array of test vectors X.
        
        Parameters:
            X (np.ndarray): The input samples. An array of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: The predicted classes for each input sample in X.
        '''
        # Calculate the log probabilities of each class for the input samples
        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        # Predict the class with the highest probability
        return self.classes_[np.argmax(log_probs, axis=1)]

# Example usage (commented out to prevent accidental execution):
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X_train, y_train)
# y_pred = nb_classifier.predict(X_test)
# print(y_pred[:5])
