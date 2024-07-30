import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Concatenate, Dropout
from tensorflow.keras.models import Model
from advance_preprocessing import load_and_preprocess_data, clean_text
from train_advance_models import train_bilstm, train_bert_cnn, train_distilbert

# Initialize DistilBERT tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 64  # or whatever maximum length you want to use

def ensure_dir(directory):
    """
    Ensures that the specified directory exists. If it does not exist, creates it.

    Parameters:
    directory (str): The directory path to ensure existence.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    """
    Plots the confusion matrix for the given true and predicted labels and saves the plot as an image file.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    labels (list): List of label names.
    model_name (str): Name of the model for labeling the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    plt.tight_layout()
    
    ensure_dir('outputs')
    plt.savefig(f'outputs/confusion_matrix_{model_name}.png')
    plt.close()

def plot_training_history(history, model_name):
    """
    Plots the training history (accuracy and loss) and saves the plot as an image file.

    Parameters:
    history (History): Keras History object containing training history.
    model_name (str): Name of the model for labeling the plot.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    
    ensure_dir('outputs')
    plt.savefig(f'outputs/training_history_{model_name}.png')
    plt.close()

def clean_and_prepare_df(df):
    """
    Cleans and prepares the dataframe by removing labels with only one occurrence and applying text cleaning.

    Parameters:
    df (DataFrame): The input dataframe containing 'Emotion Label' and 'Text' columns.

    Returns:
    DataFrame: The cleaned and prepared dataframe.
    """
    label_counts = df['Emotion Label'].value_counts()
    labels_to_remove = label_counts[label_counts == 1].index
    df = df[~df['Emotion Label'].isin(labels_to_remove)]
    df = df[["Emotion Label", "Text"]]
    df = df.dropna()

    df['Text'] = df['Text'].apply(clean_text)
    print(df.shape)
    return df

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates the model on the test data and generates a confusion matrix, classification report, and accuracy/f1 scores.

    Parameters:
    model (Model): The trained model to evaluate.
    X_test (array-like): The test input data.
    y_test (array-like): The true labels for the test data.
    model_name (str): The name of the model for labeling outputs.

    Returns:
    tuple: The predicted class labels, accuracy, and F1 score.
    """
    if model_name in ['DistilBERT', 'BERT-CNN']:
        y_pred = model.predict([X_test['input_ids'], X_test['attention_mask']])
    else:
        y_pred = model.predict(X_test)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    report = classification_report(y_true, y_pred_classes)

    labels = ['joy', 'guilt', 'sadness', 'disgust', 'shame', 'anger', 'fear']
    plot_confusion_matrix(y_true, y_pred_classes, labels, model_name)

    ensure_dir('outputs')
    with open(f'outputs/evaluation_{model_name}.txt', 'w') as f:
        f.write(f"Evaluation results for {model_name}:\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    return y_pred_classes, accuracy, f1

def save_predictions(y_pred, dataset_name, model_name):
    """
    Saves the predicted class labels to a text file.

    Parameters:
    y_pred (array-like): The predicted class labels.
    dataset_name (str): The name of the dataset (e.g., 'test').
    model_name (str): The name of the model for labeling the output file.
    """
    ensure_dir('outputs')
    np.savetxt(f'outputs/predictions_{dataset_name}_{model_name}.txt', y_pred, fmt='%d')

def encode_texts(texts, tokenizer, max_len):
    """
    Encodes the input texts using the specified tokenizer and maximum length.

    Parameters:
    texts (list of str): The input texts to encode.
    tokenizer (Tokenizer): The tokenizer to use for encoding.
    max_len (int): The maximum length of the encoded sequences.

    Returns:
    dict: A dictionary containing encoded input IDs and attention masks.
    """
    return tokenizer(texts, max_length=max_len, padding=True, truncation=True, return_tensors="tf")

def train_and_evaluate_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test):
    """
    Trains and evaluates the specified model on the provided data.

    Parameters:
    model_name (str): The name of the model to train and evaluate ('BiLSTM', 'DistilBERT', 'BERT-CNN').
    X_train (array-like): The training input data.
    y_train (array-like): The training labels.
    X_val (array-like): The validation input data.
    y_val (array-like): The validation labels.
    X_test (array-like): The test input data.
    y_test (array-like): The test labels.
    df_train (DataFrame): The original training dataframe.
    df_val (DataFrame): The original validation dataframe.
    df_test (DataFrame): The original test dataframe.

    Returns:
    tuple: The accuracy and F1 score of the model on the test set.
    """
    if model_name == 'BiLSTM':
        model, history = train_bilstm(X_train, y_train, X_val, y_val)
        X_test_eval = X_test
    elif model_name in ['DistilBERT', 'BERT-CNN']:
        # Clean and prepare dataframes
        df_train_clean = clean_and_prepare_df(df_train)
        df_val_clean = clean_and_prepare_df(df_val)
        df_test_clean = clean_and_prepare_df(df_test)
        
        # Debug prints
        print(f"Type of df_train_clean['Text']: {type(df_train_clean['Text'])}")
        print(f"First few items of df_train_clean['Text']: {df_train_clean['Text'].head().tolist()}")
        
        # Encode cleaned texts
        X_train_encoded = encode_texts(df_train_clean['Text'].tolist(), distilbert_tokenizer, MAX_LEN)
        X_val_encoded = encode_texts(df_val_clean['Text'].tolist(), distilbert_tokenizer, MAX_LEN)
        X_test_encoded = encode_texts(df_test_clean['Text'].tolist(), distilbert_tokenizer, MAX_LEN)
        
        # Prepare labels
        y_train_clean = tf.keras.utils.to_categorical(df_train_clean['Emotion Label'].astype('category').cat.codes)
        y_val_clean = tf.keras.utils.to_categorical(df_val_clean['Emotion Label'].astype('category').cat.codes)
        y_test_clean = tf.keras.utils.to_categorical(df_test_clean['Emotion Label'].astype('category').cat.codes)
        
        if model_name == 'DistilBERT':
            model, history = train_distilbert(X_train_encoded, y_train_clean, X_val_encoded, y_val_clean)
        else:  # BERT-CNN
            model, history = train_bert_cnn(X_train_encoded, y_train_clean, X_val_encoded, y_val_clean)
        
        X_test_eval = X_test_encoded
        y_test = y_test_clean
    else:
        raise ValueError(f"Unknown model: {model_name}")

    plot_training_history(history, model_name)
    
    y_pred, accuracy, f1 = evaluate_model(model, X_test_eval, y_test, model_name)
    save_predictions(y_pred, 'test', model_name)

    print(f"{model_name} - Final Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-----------------------------")

    return accuracy, f1

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    train_path = base_dir / 'data' / 'isear' / 'isear-train.csv'
    val_path = base_dir / 'data' / 'isear' / 'isear-val.csv'
    test_path = base_dir / 'data' / 'isear' / 'isear-test.csv'

    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = load_and_preprocess_data(
        str(train_path), str(val_path), str(test_path))

    # Load original dataframes
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    models = ['BiLSTM', 'DistilBERT', 'BERT-CNN']
    results = {}

    for model_name in models:
        print(f"Training and evaluating {model_name}...")
        accuracy, f1 = train_and_evaluate_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test)
        results[model_name] = {'Accuracy': accuracy, 'F1-Score': f1}
        print(f"Completed {model_name}\n")

    print("\nSummary of all models:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
        print("-----------------------------")

    print("All models trained and evaluated. Results saved in the 'outputs' directory.")