import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling1D
from tensorflow.keras.models import Model

class DistilBERTModel:
    def __init__(self, max_len, num_classes):
        self.max_len = max_len
        self.num_classes = num_classes
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.base_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model = self._build_model()

    def _build_model(self):
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name='attention_mask')

        distilbert_output = self.base_model(input_ids, attention_mask=attention_mask)[0]
        pooled_output = GlobalMaxPooling1D()(distilbert_output)
        dense = Dense(256, activation='relu')(pooled_output)
        output = Dense(self.num_classes, activation='softmax')(dense)

        model = Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, texts, labels, epochs, batch_size, validation_split=0.1):
        encodings = self.tokenizer(texts, max_length=self.max_len, padding=True, truncation=True, return_tensors="tf")
        return self.model.fit(
            [encodings['input_ids'], encodings['attention_mask']],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def evaluate(self, texts, labels):
        encodings = self.tokenizer(texts, max_length=self.max_len, padding=True, truncation=True, return_tensors="tf")
        return self.model.evaluate([encodings['input_ids'], encodings['attention_mask']], labels)

    def predict(self, texts):
        encodings = self.tokenizer(texts, max_length=self.max_len, padding=True, truncation=True, return_tensors="tf")
        return self.model.predict([encodings['input_ids'], encodings['attention_mask']])