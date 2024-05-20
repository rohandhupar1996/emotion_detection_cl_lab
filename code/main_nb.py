''' 
Project: Multinomial Naive Bayes Classifier for Emotion Detections
Version: 0.1
Author: Rohan Dhupar

Description:
This script integrates preprocessing, model training, and evaluation of a Multinomial Naive Bayes classifier
for emotion detection from text data. It uses custom classes for preprocessing the text, performing the naive Bayes
classification, and evaluating the model's performance on different datasets.
'''

# Import necessary custom classes for Preprocess, Naive Bayes, and Evaluation
from preprocess import SimpleCountVectorizer, train_test_split
from modified_nb import MultinomialNB
from evaluation import EmotionEvaluation
from dummy_evaluation import random_prediction, majority_vote_prediction
import pandas as pd
import numpy as np

# File paths for train, test, and validation datasets
train = '../data/isear/isear-train.csv'
test = '../data/isear/isear-test.csv'
val = '../data/isear/isear-val.csv'

def main():
    '''
    Main function to handle model training, testing, and validation along with dummy validations.
    It processes the data, trains a Multinomial Naive Bayes classifier, evaluates its performance,
    and prints the results for training, testing, and validation sets.
    '''
    # Initialize and process training data
    simple_vectorizer = SimpleCountVectorizer(max_features=3000)
    data = pd.read_csv(train)
    cleaned_data = data.dropna(subset=['Text', 'Emotion Label'])
    X = simple_vectorizer.fit_transform(cleaned_data['Text'].values)
    y = cleaned_data['Emotion Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Training process
    print("################################## Training ########################################################\n")
    print("Training...\n")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    # Predict on test data and evaluate
    y_pred = nb_model.predict(X_test)
    emotions = ['joy', 'guilt', 'sadness', 'shame', 'disgust', 'anger', 'fear']
    evaluation = EmotionEvaluation(y_test, y_pred, emotions)
    evaluation.compute_metrics()
    final_scores = evaluation.calculate_precision_recall_f1()
    average_f1_scores = evaluation.calculate_average_f1()
    print("Final Scores by Emotion on training set:")
    for emotion, scores in final_scores.items():
        print(f"{emotion} - Precision: {scores['Precision']:.2f}, Recall: {scores['Recall']:.2f}, F1 Score: {scores['F1 Score']:.2f}")

    print("\nAverage F1 Scores:", average_f1_scores)

    # Repeat the process for validation and testing data sets
    for phase in ['Testing', 'Validation']:
        data_path = val if phase == 'Validation' else test
        data = pd.read_csv(data_path)
        cleaned_data = data.dropna(subset=['Text', 'Emotion Label'])
        X = simple_vectorizer.transform(cleaned_data['Text'].values)
        y_test = cleaned_data['Emotion Label'].values
        y_pred = nb_model.predict(X)

        evaluation = EmotionEvaluation(y_test, y_pred, emotions)
        evaluation.compute_metrics()
        final_scores = evaluation.calculate_precision_recall_f1()
        average_f1_scores = evaluation.calculate_average_f1()
        
        print(f"################################## {phase} ########################################################\n")
        print(f"Final Scores by Emotion on {phase.lower()} set:")
        for emotion, scores in final_scores.items():
            print(f"{emotion} - Precision: {scores['Precision']:.2f}, Recall: {scores['Recall']:.2f}, F1 Score: {scores['F1 Score']:.2f}")

        print(f"\nAverage F1 Scores on {phase.lower()} set:", average_f1_scores)

    # Dummy validation
    print("\n################################## Dummy Validation ########################################################\n")
    # Example data setup and predictions using dummy methods
    y_true = ['joy', 'joy', 'sadness', 'fear', 'disgust', 'joy', 'sadness', 'anger', 'disgust', 'fear']
    most_common_class = max(set(y_true), key=y_true.count)
    y_pred_random = random_prediction(emotions, len(y_true))
    y_pred_majority = majority_vote_prediction(most_common_class, len(y_true))

    # Evaluation of dummy methods
    evaluation_random = EmotionEvaluation(y_true, y_pred_random, emotions)
    evaluation_majority = EmotionEvaluation(y_true, y_pred_majority, emotions)
    evaluation_random.compute_metrics()
    evaluation_majority.compute_metrics()
    final_scores_random = evaluation_random.calculate_precision_recall_f1()
    final_scores_majority = evaluation_majority.calculate_precision_recall_f1()
    average_f1_scores_random = evaluation_random.calculate_average_f1()
    average_f1_scores_majority = evaluation_majority.calculate_average_f1()

    print("Random Prediction Evaluation:", final_scores_random)
    print("Average F1 Scores (Random):", average_f1_scores_random)
    print("Majority Vote Prediction Evaluation:", final_scores_majority)
    print("Average F1 Scores (Majority Vote):", average_f1_scores_majority)

if __name__ == '__main__':
    main()
