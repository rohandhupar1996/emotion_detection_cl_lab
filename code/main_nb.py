#Import our custom classes for Preprocess, Naive Bayes, and Evaluation

from preprocess import TextDataProcessor
from modified_nb import MultiLabelNaiveBayes
from evaluation import EmotionEvaluation
import pandas as pd

train = '../data/ssec/train.csv'
test = '../data/ssec/test.csv' 
dev = '../data/val/val.csv'


# main.py

import sys
from preprocess import your_preprocessing_function
from feature_extractor import your_feature_extraction_function
from modified_nb import your_modified_nb_function
from evaluation import your_evaluation_function

def main():
    # Train
    processor = TextDataProcessor(train)
    # Run the processing methods
    processor.load_data()
    processor.preprocess_text()
    processor.convert_emotions_to_binary()
    combined_data = processor.get_data()
    # Split the data into features and target
    X_train = combined_data.iloc[:, :-8] 
    y_train = combined_data.iloc[:, -8:]
        
    
    nb_model=MultiLabelNaiveBayes()
    nb_model.train(X_train, y_train)
    
    # Test 
    processor = TextDataProcessor(test)
    # Run the processing methods
    processor.load_data()
    processor.preprocess_text()
    processor.convert_emotions_to_binary()
    combined_data = processor.get_data()
    X_test = combined_data.iloc[:, :-8] 
    y_true = combined_data.iloc[:, -8:]

    y_pred = []
    for item in X_test:
        pred = nb_model.predict(item)
        y_pred.append(pred)
        
    ev=EmotionEvaluation()
    emotions = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
    evaluation = EmotionEvaluation(y_true, y_pred, emotions)  # y_true and y_pred need to be lists of dictionaries
    evaluation.compute_metrics()
    final_scores = evaluation.calculate_precision_recall_f1()
    average_f1_scores = evaluation.calculate_average_f1()
    print(final_scores)
    print("Average F1 Scores:", average_f1_scores)
        

if __name__ == '__main__':
    main()


