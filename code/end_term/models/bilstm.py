from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

class BiLSTMModel:
    def __init__(self, vocab_size, max_len, num_classes):
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
            Bidirectional(LSTM(100)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, Y, epochs, batch_size, validation_data=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return self.model.predict(X)