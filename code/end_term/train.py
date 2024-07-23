from preprocessing import load_and_preprocess_data
from models.bilstm import BiLSTMModel
from models.distilbert import DistilBERTModel
from models.bert_cnn import BERTCNNModel
import pandas as pd

def train_bilstm():
    X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_data('data/train.csv', 'data/test.csv')
    vocab_size = len(tokenizer.word_index) + 1
    max_len = X_train.shape[1]
    num_classes = y_train.shape[1]

    model = BiLSTMModel(vocab_size, max_len, num_classes)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return model, history

def train_distilbert():
    _, y_train, _, y_test, _ = load_and_preprocess_data('data/train.csv', 'data/test.csv')
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    
    max_len = 64
    num_classes = y_train.shape[1]

    model = DistilBERTModel(max_len, num_classes)
    history = model.fit(df_train['Text'].tolist(), y_train, epochs=3, batch_size=16, validation_split=0.1)
    return model, history

def train_bert_cnn():
    _, y_train, _, y_test, _ = load_and_preprocess_data('data/train.csv', 'data/test.csv')
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    
    max_len = 128
    num_classes = y_train.shape[1]

    model = BERTCNNModel(max_len, num_classes)
    history = model.fit(df_train['Text'].tolist(), y_train, epochs=5, batch_size=16, validation_split=0.1)
    return model, history

if __name__ == "__main__":
    bilstm_model, bilstm_history = train_bilstm()
    distilbert_model, distilbert_history = train_distilbert()
    bert_cnn_model, bert_cnn_history = train_bert_cnn()