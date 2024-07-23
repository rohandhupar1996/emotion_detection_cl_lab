import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(train_path, test_path, max_len=100, num_words=10000):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Clean and preprocess text
    df_train['Text'] = df_train['Text'].apply(clean_text)
    df_test['Text'] = df_test['Text'].apply(clean_text)

    # Encode labels
    predefined_labels = {
        'joy': 0, 'guilt': 1, 'sadness': 2, 'disgust': 3,
        'shame': 4, 'anger': 5, 'fear': 6
    }
    df_train['Emotion Label Encoded'] = df_train['Emotion Label'].map(predefined_labels)
    df_test['Emotion Label Encoded'] = df_test['Emotion Label'].map(predefined_labels)

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df_train["Text"])
    X_train_seq = tokenizer.texts_to_sequences(df_train["Text"])
    X_test_seq = tokenizer.texts_to_sequences(df_test["Text"])
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

    # Convert labels to categorical
    num_classes = len(predefined_labels)
    y_train_categorical = to_categorical(df_train["Emotion Label Encoded"], num_classes=num_classes)
    y_test_categorical = to_categorical(df_test["Emotion Label Encoded"], num_classes=num_classes)

    return X_train_padded, y_train_categorical, X_test_padded, y_test_categorical, tokenizer