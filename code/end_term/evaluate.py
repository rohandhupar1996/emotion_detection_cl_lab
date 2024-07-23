import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd
from preprocessing import load_and_preprocess_data
from train import train_bilstm, train_distilbert, train_bert_cnn

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    if isinstance(X_test[0], str):  # Check if X_test is a list of strings
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(f"Evaluation results for {model_name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred_classes))
    print("F1 Score:", f1_score(y_true, y_pred_classes, average='weighted'))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes))

    labels = ['joy', 'guilt', 'sadness', 'disgust', 'shame', 'anger', 'fear']
    plot_confusion_matrix(y_true, y_pred_classes, labels)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _ = load_and_preprocess_data('data/isear-train.csv', 'data/isear-test.csv')

    bilstm_model, bilstm_history = train_bilstm()
    evaluate_model(bilstm_model, X_test, y_test, "BiLSTM")
    plot_training_history(bilstm_history)

    distilbert_model, distilbert_history = train_distilbert()
    df_test = pd.read_csv('data/isear-test.csv')
    evaluate_model(distilbert_model, df_test['Text'].tolist(), y_test, "DistilBERT")
    plot_training_history(distilbert_history)

    bert_cnn_model, bert_cnn_history = train_bert_cnn()
    evaluate_model(bert_cnn_model, df_test['Text'].tolist(), y_test, "BERT-CNN")
    plot_training_history(bert_cnn_history)