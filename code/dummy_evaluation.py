import numpy as np

def random_prediction(labels, size):
    return np.random.choice(labels, size=size)

def majority_vote_prediction(most_common_label, size):
    return [most_common_label] * size
