import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertTokenizer

class BERTCNNModel:
    def __init__(self, max_len, num_classes):
        self.max_len = max_len
        self.num_classes = num_classes
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.base_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.model = self._build_model()

    def _build_model(self):
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name='attention_mask')

        bert_output = self.base_model(input_ids, attention_mask=attention_mask)[0]

        conv1 = Conv1D(128, 2, activation='relu')(bert_output)
        pool1 = GlobalMaxPooling1D()(conv1)

        conv2 = Conv1D(128, 3, activation='relu')(bert_output)
        pool2 = GlobalMaxPooling1D()(conv2)

        conv3 = Conv1D(128, 4, activation='relu')(bert_output)
        pool3 = GlobalMaxPooling1D()(conv3)

        concatenated = Concatenate()([pool1, pool2, pool3])
        dense = Dense(256, activation='relu')(concatenated)
        output = Dense(self.num_classes, activation='softmax')(dense)

        model = Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, texts, labels, epochs, batch_size, validation_split=0.1):
        encodings = self.tokenizer(texts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="tf")
        return self.model.fit(
            [encodings['input_ids'], encodings['attention_mask']],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def evaluate(self, texts, labels):
        encodings = self.tokenizer(texts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="tf")
        return self.model.evaluate([encodings['input_ids'], encodings['attention_mask']], labels)

    def predict(self, texts):
        encodings = self.tokenizer(texts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="tf")
        return self.model.predict([encodings['input_ids'], encodings['attention_mask']])