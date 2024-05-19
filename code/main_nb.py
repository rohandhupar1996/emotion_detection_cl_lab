# Import necessary custom classes for Preprocess, Naive Bayes, and Evaluation
from preprocess import SimpleCountVectorizer, train_test_split
from modified_nb import MultinomialNB
from evaluation import EmotionEvaluation
from dummy_evaluation import random_prediction , majority_vote_prediction
import pandas as pd
import numpy as np


# File paths for train, test, and validation datasets
train = '../data/isear/isear-train.csv'
test = '../data/isear/isear-test.csv' 
val = '../data/isear/isear-val.csv'

def main():
    # Initialize and process training data
    simple_vectorizer = SimpleCountVectorizer(max_features=3000)
    data=pd.read_csv(train)
    cleaned_data = data.dropna(subset=['Text', 'Emotion Label'])
    X = simple_vectorizer.fit_transform(cleaned_data['Text'].values)
    y = cleaned_data['Emotion Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(X_train.shape,X_train)
    print("################################## Training ########################################################")
    print("\n")
    print("Training...")
    print("\n")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    # # Predict on test data
    y_pred = nb_model.predict(X_test)
    print(y_pred)
    emotions = ['joy', 'guilt', 'sadness', 'shame', 'disgust', 'anger', 'fear']
    evaluation = EmotionEvaluation(y_test, y_pred, emotions)  # y_true and y_pred are lists of emotion labels
    evaluation.compute_metrics()
    final_scores = evaluation.calculate_precision_recall_f1()
    average_f1_scores = evaluation.calculate_average_f1()
    print(final_scores)
    print("Average F1 Scores:", average_f1_scores)

    print("Final Scores by Emotion on training set:")
    for emotion, scores in final_scores.items():
        print(f"{emotion} - Precision: {scores['Precision']:.2f}, Recall: {scores['Recall']:.2f}, F1 Score: {scores['F1 Score']:.2f}")

    print("\nAverage F1 Scores:", average_f1_scores)
    print("\n")
    print("################################## Testing ########################################################")
    print("\n")
    print("Testing")
    print("\n")
    data=pd.read_csv(val)
    cleaned_data = data.dropna(subset=['Text', 'Emotion Label'])
    X = simple_vectorizer.fit_transform(cleaned_data['Text'].values)
    y_test = cleaned_data['Emotion Label'].values
    y_pred = nb_model.predict(X)
    
    evaluation = EmotionEvaluation(y_test, y_pred, emotions)  # y_true and y_pred are lists of emotion labels
    evaluation.compute_metrics()
    final_scores = evaluation.calculate_precision_recall_f1()
    average_f1_scores = evaluation.calculate_average_f1()
    
    print(final_scores)
    
    print("Final Scores by Emotion on testing set:")
    for emotion, scores in final_scores.items():
        print(f"{emotion} - Precision: {scores['Precision']:.2f}, Recall: {scores['Recall']:.2f}, F1 Score: {scores['F1 Score']:.2f}")

    print("\nAverage F1 Scores on testing set::", average_f1_scores)
    
    print("\n")
    print("################################## validation ########################################################")
    print("\n")
    print("validation")
    print("\n")
    data=pd.read_csv(val)
    cleaned_data = data.dropna(subset=['Text', 'Emotion Label'])
    X = simple_vectorizer.fit_transform(cleaned_data['Text'].values)
    y_test = cleaned_data['Emotion Label'].values
    y_pred = nb_model.predict(X)
    
    evaluation = EmotionEvaluation(y_test, y_pred, emotions)  # y_true and y_pred are lists of emotion labels
    evaluation.compute_metrics()
    final_scores = evaluation.calculate_precision_recall_f1()
    average_f1_scores = evaluation.calculate_average_f1()
    
    print(final_scores)
    print("\n")
    
    print("Final Scores by Emotion on validation set:")
    for emotion, scores in final_scores.items():
        print(f"{emotion} - Precision: {scores['Precision']:.2f}, Recall: {scores['Recall']:.2f}, F1 Score: {scores['F1 Score']:.2f}")

    print("\nAverage F1 Scores on validation set::", average_f1_scores)
    
    
    
    print("\n")
    print("################################## dummy validation ########################################################")
    print("\n")
    print("dummy validation")
    
    # Example data setup
    y_true = ['joy', 'Joy', 'sadness', 'fear', 'disgust', 'joy', 'sadness', 'anger', 'disgust', 'fear']
    emotions = ['joy', 'guilt', 'sadness', 'shame', 'disgust', 'anger', 'fear']

    # Find the most common class
    most_common_class = max(set(y_true), key=y_true.count)

    # Generate predictions using the dummy methods
    y_pred_random = random_prediction(emotions, len(y_true))
    y_pred_majority = majority_vote_prediction(most_common_class, len(y_true))
    
    
    # Initialize the EmotionEvaluation class and compute metrics
    evaluation_random = EmotionEvaluation(y_true, y_pred_random, emotions)
    evaluation_majority = EmotionEvaluation(y_true, y_pred_majority, emotions)

    # Compute metrics
    evaluation_random.compute_metrics()
    evaluation_majority.compute_metrics()

    # Get precision, recall, F1-score for each method
    final_scores_random = evaluation_random.calculate_precision_recall_f1()
    final_scores_majority = evaluation_majority.calculate_precision_recall_f1()

    # Calculate average F1 scores
    average_f1_scores_random = evaluation_random.calculate_average_f1()
    average_f1_scores_majority = evaluation_majority.calculate_average_f1()

    # Print the results
    print("Random Prediction Evaluation:", final_scores_random)
    print("Average F1 Scores (Random):", average_f1_scores_random)

    print("Majority Vote Prediction Evaluation:", final_scores_majority)
    print("Average F1 Scores (Majority Vote):", average_f1_scores_majority)

    
    
    
if __name__ == '__main__':
    main()
