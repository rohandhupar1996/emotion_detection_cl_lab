''' 
Project: Multinomial Naive Bayes Classifier for Emotion Detection
Version: 0.1
Author: Rohan Dhupar

Description:
This script defines functions to simulate predictions of emotions using a simple approach before
implementing a full multinomial Naive Bayes classifier. It includes functions for random predictions
and predictions based on majority voting.

'''

import numpy as np  # Importing the numpy library for numerical operations

def random_prediction(labels, size):
    '''
    Simulate random predictions for emotion detection.
    
    Parameters:
        labels (list): A list of possible emotion labels.
        size (int): The number of predictions to generate.
        
    Returns:
        list: A list containing randomly chosen emotion labels.
    '''
    return np.random.choice(labels, size=size)

def majority_vote_prediction(most_common_label, size):
    '''
    Generates predictions based on the most common label (majority vote).
    
    Parameters:
        most_common_label (str): The label that occurs most frequently.
        size (int): The number of predictions to generate.
        
    Returns:
        list: A list where every element is the most common label.
    '''
    return [most_common_label] * size
