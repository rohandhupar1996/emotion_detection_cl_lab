import sys
import time

class MultiLabelNaiveBayes:
    def __init__(self):
        self.models = {}

    def train(self, X, Y):
        print("Training start...")
        for column in Y.columns:
            self.animate_training(column)
            self.models[column] = self.train_single_label(X, Y[column])
        print("Training completed!")

    def train_single_label(self, X, y):
        class_count = Counter(y)
        doc_count = len(y)
        prior_probabilities = {cls: math.log(class_count[cls] / doc_count) for cls in class_count}
        feature_counts = {}
        for cls in class_count:
            feature_counts[cls] = Counter()
        for text, label in zip(X, y):
            feature_counts[label].update(text.split())
        feature_probabilities = {}
        for cls in class_count:
            total_features = sum(feature_counts[cls].values()) + len(feature_counts[cls])
            feature_probabilities[cls] = {word: math.log((feature_counts[cls].get(word, 0) + 1) / total_features)
                                          for word in feature_counts[cls]}
        return {'prior': prior_probabilities, 'likelihood': feature_probabilities}

    def predict(self, X):
        results = []
        for text in X:
            result = {}
            for emotion, model in self.models.items():
                log_prob = {cls: model['prior'][cls] for cls in model['prior']}
                words = text.split()
                for word in words:
                    for cls in model['likelihood']:
                        log_prob[cls] += model['likelihood'][cls].get(word, 0)
                predicted_class = max(log_prob, key=log_prob.get)
                result[emotion] = predicted_class
            results.append(result)
        return results

    def animate_training(self, label):
        animation = "|/-\\"
        for i in range(10):  # Adjust the range for longer display if needed
            time.sleep(0.1)  # Speed of animation
            sys.stdout.write(f"\rTraining {label} " + animation[i % len(animation)])
            sys.stdout.flush()
        print()  # Move to new line

# Example usage
# nb_model = MultiLabelNaiveBayes()
# nb_model.train(X_train, Y_train)  # Define X_train and Y_train appropriately
# predictions = nb_model.predict(X_test)  # Define X_test appropriately
