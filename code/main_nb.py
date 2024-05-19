#Import our custom classes for Preprocess, Naive Bayes, and Evaluation

from preprocess import TextDataProcessor
from modified_nb import MultiLabelNaiveBayes
from evaluation import EmotionEvaluation
import pandas as pd

train = '../data/ssec/train.csv'
test = '../data/ssec/test.csv' 
dev = '../data/val/val.csv'


# main.py
from preprocess import TextDataProcessor
from modified_nb import MultiLabelNaiveBayes
from evaluation import EmotionEvaluation

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
        
    print("train")
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
        
    # List of emotions you are evaluating
    emotions = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

    # Create an instance of EmotionEvaluation with the appropriate data
    evaluation = EmotionEvaluation(y_true, y_pred, emotions)

    # Compute metrics
    evaluation.compute_metrics()

    # Calculate precision, recall, and F1 score for each emotion
    final_scores = evaluation.calculate_precision_recall_f1()

    # Calculate average F1 scores
    average_f1_scores = evaluation.calculate_average_f1()

    # Print the results
    print("Final Scores by Emotion:")
    for emotion, scores in final_scores.items():
        print(f"{emotion} - Precision: {scores['Precision']:.2f}, Recall: {scores['Recall']:.2f}, F1 Score: {scores['F1 Score']:.2f}")

    print("\nAverage F1 Scores:", average_f1_scores)

        

if __name__ == '__main__':
    main()


