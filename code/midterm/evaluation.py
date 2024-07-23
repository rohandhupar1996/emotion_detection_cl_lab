''' 
Project: Multinomial Naive Bayes Classifier for Emotion Detection
Version: 0.1
Author: Rohan Dhupar

Description:
This module defines a class for evaluating emotion detection models. It includes methods for computing confusion matrices
for each emotion, calculating precision, recall, F1 score for individual emotions, and averaging these scores across emotions.
'''

from typing import List, Any
from collections import defaultdict

class EmotionEvaluation:
    '''
    A class to evaluate emotion detection using various metrics like precision, recall, and F1 score.
    
    Attributes:
        y_true (List[Any]): Actual emotion labels.
        y_pred (List[Any]): Predicted emotion labels.
        emotions (List[str]): List of all possible emotions.
        confusion_matrices (dict): A dictionary to hold confusion matrices for each emotion.
    '''

    def __init__(self, y_true: List[Any], y_pred: List[Any], emotions: List[str]):
        '''
        Initializes the EmotionEvaluation with the true labels, predicted labels, and possible emotions.
        
        Parameters:
            y_true (List[Any]): The actual labels.
            y_pred (List[Any]): The predicted labels.
            emotions (List[str]): The list of all possible emotions.
        '''
        self.y_true = y_true
        self.y_pred = y_pred
        self.emotions = emotions
        self.confusion_matrices = {emotion: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for emotion in emotions}

    def compute_metrics(self):
        '''
        Computes the confusion matrix for each emotion.
        '''
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            for emotion in self.emotions:
                if true_label == emotion and pred_label == emotion:
                    self.confusion_matrices[emotion]['TP'] += 1
                elif true_label == emotion and pred_label != emotion:
                    self.confusion_matrices[emotion]['FN'] += 1
                elif true_label != emotion and pred_label == emotion:
                    self.confusion_matrices[emotion]['FP'] += 1
                elif true_label != emotion and pred_label != emotion:
                    self.confusion_matrices[emotion]['TN'] += 1

    def calculate_precision_recall_f1(self):
        '''
        Calculates precision, recall, and F1 score for each emotion based on the confusion matrices.
        
        Returns:
            dict: A dictionary containing precision, recall, and F1 score for each emotion.
        '''
        results = {}
        for emotion in self.emotions:
            TP = self.confusion_matrices[emotion]['TP']
            FP = self.confusion_matrices[emotion]['FP']
            FN = self.confusion_matrices[emotion]['FN']
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            results[emotion] = {'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        return results

    def calculate_average_f1(self):
        '''
        Calculates macro-average and micro-average F1 scores across all emotions.
        
        Returns:
            dict: A dictionary containing both macro-average and micro-average F1 scores.
        '''
        f1_scores = [self.calculate_precision_recall_f1()[emotion]['F1 Score'] for emotion in self.emotions]
        macro_f1 = sum(f1_scores) / len(f1_scores)  # Macro-average F1
        total_TP = sum([self.confusion_matrices[emotion]['TP'] for emotion in self.emotions])
        total_FP = sum([self.confusion_matrices[emotion]['FP'] for emotion in self.emotions])
        total_FN = sum([self.confusion_matrices[emotion]['FN'] for emotion in self.emotions])
        total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        micro_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        return {'Macro F1': macro_f1, 'Micro F1': micro_f1}

# Example usage:
# emotions = ['Anger', 'Joy', 'Sadness', 'Fear', 'Disgust']
# evaluation = Em
