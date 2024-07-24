import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from transformers import TFDistilBertModel, DistilBertTokenizer
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model

def train_bilstm(X_train, y_train, X_val, y_val):
    vocab_size = X_train.shape[1]
    max_len = X_train.shape[1]
    num_classes = y_train.shape[1]

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(100)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model, history


def train_distilbert(train_texts, y_train, val_texts, y_val):
    max_len = 64
    batch_size = 16
    num_classes = y_train.shape[1]

    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    distilbert_output = distilbert_model(input_ids, attention_mask=attention_mask)[0]
    pooled_output = GlobalMaxPooling1D()(distilbert_output)
    dense = Dense(256, activation='relu')(pooled_output)
    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        [train_texts.input_ids, train_texts.attention_mask],
        y_train,
        validation_data=([val_texts.input_ids, val_texts.attention_mask], y_val),
        epochs=3,
        batch_size=batch_size
    )

    return model, history

def train_bert_cnn(train_texts, y_train, val_texts, y_val):
    max_len = 64
    batch_size = 16
    num_classes = y_train.shape[1]

    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

    conv1 = Conv1D(128, 2, activation='relu')(bert_output)
    pool1 = GlobalMaxPooling1D()(conv1)

    conv2 = Conv1D(128, 3, activation='relu')(bert_output)
    pool2 = GlobalMaxPooling1D()(conv2)

    conv3 = Conv1D(128, 4, activation='relu')(bert_output)
    pool3 = GlobalMaxPooling1D()(conv3)

    concatenated = Concatenate()([pool1, pool2, pool3])
    dense = Dense(256, activation='relu')(concatenated)
    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        [train_texts.input_ids, train_texts.attention_mask],
        y_train,
        validation_data=([val_texts.input_ids, val_texts.attention_mask], y_val),
        epochs=3,
        batch_size=batch_size
    )

    return model, history