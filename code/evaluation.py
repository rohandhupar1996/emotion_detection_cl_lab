from typing_extensions import List , Any

class EmotionEvaluation:
    def __init__(self, y_true: List[Any], y_pred: List[Any], emotions: List[str]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.emotions = emotions
        self.confusion_matrices = {emotion: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for emotion in emotions}

    def compute_metrics(self):
        for true_labels, pred_labels in zip(self.y_true, self.y_pred):
            for emotion in self.emotions:
                true = true_labels.get(emotion, 0)
                pred = pred_labels.get(emotion, 0)
                if true == 1 and pred == 1:
                    self.confusion_matrices[emotion]['TP'] += 1
                elif true == 1 and pred == 0:
                    self.confusion_matrices[emotion]['FN'] += 1
                elif true == 0 and pred == 1:
                    self.confusion_matrices[emotion]['FP'] += 1
                elif true == 0 and pred == 0:
                    self.confusion_matrices[emotion]['TN'] += 1

    def calculate_precision_recall_f1(self):
        results = {}
        for emotion in self.emotions:
            TP = self.confusion_matrices[emotion]['TP']
            FP = self.confusion_matrices[emotion]['FP']
            TN = self.confusion_matrices[emotion]['TN']
            FN = self.confusion_matrices[emotion]['FN']
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            results[emotion] = {'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        return results

    def calculate_average_f1(self):
        f1_scores = [self.calculate_precision_recall_f1()[emotion]['F1 Score'] for emotion in self.emotions]
        macro_f1 = sum(f1_scores) / len(f1_scores)  # Macro-average F1
        # Micro-average F1
        total_TP = sum([self.confusion_matrices[emotion]['TP'] for emotion in self.emotions])
        total_FP = sum([self.confusion_matrices[emotion]['FP'] for emotion in self.emotions])
        total_FN = sum([self.confusion_matrices[emotion]['FN'] for emotion in self.emotions])
        total_precision = total_TP / (total_TP + total_FP)
        total_recall = total_TP / (total_TP + total_FN)
        micro_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)
        return {'Macro F1': macro_f1, 'Micro F1': micro_f1}

# # Example usage:
# emotions = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
# evaluation = EmotionEvaluation(y_true, y_pred, emotions)  # y_true and y_pred need to be lists of dictionaries
# evaluation.compute_metrics()
# final_scores = evaluation.calculate_precision_recall_f1()
# average_f1_scores = evaluation.calculate_average_f1()
# print(final_scores)
# print("Average F1 Scores:", average_f1_scores)
