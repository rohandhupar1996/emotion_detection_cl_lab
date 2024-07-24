import pandas as pd
import numpy as np
import re
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def clean_text(text):
    """
    Cleans a given text string by performing the following steps:
    1. Converts the text to lowercase.
    2. Removes all digits.
    3. Removes all punctuation.
    4. Replaces multiple spaces with a single space and strips leading/trailing spaces.

    Parameters:
    text (str): The input text string to be cleaned.

    Returns:
    str: The cleaned text string.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(train_path, val_path, test_path, max_len=100, num_words=10000):
    """
    Loads and preprocesses the training, validation, and test datasets.
    Steps include loading data from CSV files, cleaning text, encoding labels, tokenizing, and padding sequences.

    Parameters:
    train_path (str): The file path to the training dataset CSV file.
    val_path (str): The file path to the validation dataset CSV file.
    test_path (str): The file path to the test dataset CSV file.
    max_len (int, optional): The maximum length of the padded sequences. Default is 100.
    num_words (int, optional): The maximum number of words to keep in the tokenizer's vocabulary. Default is 10000.

    Returns:
    tuple: A tuple containing:
        - X_train_padded (np.array): The padded tokenized sequences for the training set.
        - y_train_categorical (np.array): The one-hot encoded labels for the training set.
        - X_val_padded (np.array): The padded tokenized sequences for the validation set.
        - y_val_categorical (np.array): The one-hot encoded labels for the validation set.
        - X_test_padded (np.array): The padded tokenized sequences for the test set.
        - y_test_categorical (np.array): The one-hot encoded labels for the test set.
        - tokenizer (Tokenizer): The fitted Keras tokenizer.
    """
    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    # Remove labels with only one occurrence
    for df in [df_train, df_val, df_test]:
        label_counts = df['Emotion Label'].value_counts()
        labels_to_remove = label_counts[label_counts == 1].index
        df = df[~df['Emotion Label'].isin(labels_to_remove)]
    
    # Select only 'Emotion Label' and 'Text' columns
    df_train = df_train[["Emotion Label", "Text"]]
    df_val = df_val[["Emotion Label", "Text"]]
    df_test = df_test[["Emotion Label", "Text"]]
    
    # Remove rows with NaN values
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()
    
    # Clean text
    df_train['Text'] = df_train['Text'].apply(clean_text)
    df_val['Text'] = df_val['Text'].apply(clean_text)
    df_test['Text'] = df_test['Text'].apply(clean_text)
    
    # Encode labels
    predefined_labels = {
        'joy': 0, 'guilt': 1, 'sadness': 2, 'disgust': 3,
        'shame': 4, 'anger': 5, 'fear': 6
    }
    
    df_train['Emotion Label Encoded'] = df_train['Emotion Label'].map(predefined_labels)
    df_val['Emotion Label Encoded'] = df_val['Emotion Label'].map(predefined_labels)
    df_test['Emotion Label Encoded'] = df_test['Emotion Label'].map(predefined_labels)
    
    # Remove any rows with NaN in Emotion Label Encoded
    df_train = df_train.dropna(subset=['Emotion Label Encoded'])
    df_val = df_val.dropna(subset=['Emotion Label Encoded'])
    df_test = df_test.dropna(subset=['Emotion Label Encoded'])
    
    # Convert Emotion Label Encoded to integer
    df_train['Emotion Label Encoded'] = df_train['Emotion Label Encoded'].astype(int)
    df_val['Emotion Label Encoded'] = df_val['Emotion Label Encoded'].astype(int)
    df_test['Emotion Label Encoded'] = df_test['Emotion Label Encoded'].astype(int)
    
    num_classes = len(predefined_labels)
    
    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df_train["Text"])
    
    X_train_seq = tokenizer.texts_to_sequences(df_train["Text"])
    X_val_seq = tokenizer.texts_to_sequences(df_val["Text"])
    X_test_seq = tokenizer.texts_to_sequences(df_test["Text"])
    
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Convert labels to categorical
    y_train_categorical = to_categorical(df_train["Emotion Label Encoded"], num_classes=num_classes)
    y_val_categorical = to_categorical(df_val["Emotion Label Encoded"], num_classes=num_classes)
    y_test_categorical = to_categorical(df_test["Emotion Label Encoded"], num_classes=num_classes)
    
    return (X_train_padded, y_train_categorical,
            X_val_padded, y_val_categorical,
            X_test_padded, y_test_categorical,
            tokenizer)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    train_path = base_dir / 'data' / 'isear' / 'isear-train.csv'
    val_path = base_dir / 'data' / 'isear' / 'isear-val.csv'
    test_path = base_dir / 'data' / 'isear' / 'isear-test.csv'
    
    # Print unique values in Emotion Label before processing
    df_train = pd.read_csv(train_path)
    print("Unique Emotion Labels before processing:")
    print(df_train['Emotion Label'].unique())
    
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer = load_and_preprocess_data(train_path, val_path, test_path)
    
    print("\nTraining set shape:", X_train.shape)
    print("Validation set shape:", X_val.shape)
    print("Test set shape:", X_test.shape)
    
    # Print label distribution after processing
    print("\nLabel distribution in training set:")
    label_counts = y_train.sum(axis=0)
    for i, count in enumerate(label_counts):
        print(f"Label {i}: {count}")
