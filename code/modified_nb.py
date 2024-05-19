import numpy as np

class MultinomialNB:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
    
    def fit(self, X, y):
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
        # Calculate the log probabilities of each class for the input samples
        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        # Predict the class with the highest probability
        return self.classes_[np.argmax(log_probs, axis=1)]

# # Instantiate the classifier and fit it to the training data
# nb_classifier = MultinomialNBFromScratch()
# nb_classifier.fit(X_train, y_train)

# # Predict on the test data
# y_pred = nb_classifier.predict(X_test)

# y_pred[:5]
