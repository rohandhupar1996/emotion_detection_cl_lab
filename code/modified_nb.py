from collections import Counter
import math

class MultiLabelNaiveBayes:
    def __init__(self):
        self.models = {}

    def train(self, X, Y):
        # Train a separate Naive Bayes model for each label
        for column in Y.columns:
            self.models[column] = self.train_single_label(X, Y[column])

    def train_single_label(self, X, y):
        # Frequency of each class
        class_count = Counter(y)
        doc_count = len(y)
        
        # Calculate prior probabilities
        prior_probabilities = {cls: math.log(class_count[cls] / doc_count) for cls in class_count}
        
        # Feature counting per class
        feature_counts = {}
        for cls in class_count:
            feature_counts[cls] = Counter()
        
        # Count features in each class
        for text, label in zip(X, y):
            feature_counts[label].update(text.split())  # Assuming the simplest case where features are words
        
        # Calculate feature probabilities
        feature_probabilities = {}
        for cls in class_count:
            total_features = sum(feature_counts[cls].values()) + len(feature_counts[cls])  # Laplace smoothing
            feature_probabilities[cls] = {word: math.log((feature_counts[cls].get(word, 0) + 1) / total_features) 
                                          for word in feature_counts[cls]}
        
        return {'prior': prior_probabilities, 'likelihood': feature_probabilities}

    def predict(self, X):
        results = []
        for text in X:
            result = {}
            for emotion, model in self.models.items():
                log_prob = {cls: model['prior'][cls] for cls in model['prior']}  # Start with the prior probabilities
                words = text.split()  # Split text into words
                for word in words:
                    for cls in model['likelihood']:
                        log_prob[cls] += model['likelihood'][cls].get(word, 0)  # Add the log probability of each word
                
                # Find the class with the highest log probability
                predicted_class = max(log_prob, key=log_prob.get)
                result[emotion] = predicted_class
            results.append(result)
        return results

# # Example usage
# nb_model = MultiLabelNaiveBayes()
# nb_model.train(X_train, Y_train)  # X_train and Y_train need to be defined
# predictions = nb_model.predict(X_test)  # X_test needs to be defined
