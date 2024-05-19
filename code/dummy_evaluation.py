import numpy as np
from sklearn.metrics import classification_report

def random_prediction(X, labels):
    return np.random.choice(labels, size=len(X))

def majority_vote_prediction(X, most_common_class):
    return np.array([most_common_class] * len(X))

# Example usage with a dataset
y_true = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1]  # Actual labels
labels = np.unique(y_true)  # All possible labels
most_common_class = np.bincount(y_true).argmax()  # Find the most common class

# Generate predictions
y_pred_random = random_prediction(y_true, labels)
y_pred_majority = majority_vote_prediction(y_true, most_common_class)

# Evaluate and print the results
print("Random Prediction Evaluation:")
print(classification_report(y_true, y_pred_random, target_names=['Class 0', 'Class 1']))

print("Majority Vote Prediction Evaluation:")
print(classification_report(y_true, y_pred_majority, target_names=['Class 0', 'Class 1']))
