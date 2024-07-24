from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

class BiLSTMModel:
    def __init__(self, vocab_size, max_len, num_classes):
        """
        Initializes the BiLSTMModel with the specified vocabulary size, maximum sequence length, and number of output classes.
        The model architecture includes an embedding layer, a bidirectional LSTM layer, a dropout layer, and a dense output layer.

        Parameters:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        max_len (int): The maximum length of input sequences.
        num_classes (int): The number of output classes for classification.
        """
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
            Bidirectional(LSTM(100)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, Y, epochs, batch_size, validation_data=None):
        """
        Trains the model on the provided data.

        Parameters:
        X (np.array): The input data for training.
        Y (np.array): The one-hot encoded labels for the training data.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size for training.
        validation_data (tuple, optional): Data on which to evaluate the loss and any model metrics at the end of each epoch. Should be a tuple (X_val, Y_val).

        Returns:
        History: A Keras History object containing training history.
        """
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, X, Y):
        """
        Evaluates the model on the provided data.

        Parameters:
        X (np.array): The input data for evaluation.
        Y (np.array): The one-hot encoded labels for the evaluation data.

        Returns:
        tuple: The loss and accuracy of the model on the evaluation data.
        """
        return self.model.evaluate(X, Y)

    def predict(self, X):
        """
        Predicts the labels for the provided data.

        Parameters:
        X (np.array): The input data for prediction.

        Returns:
        np.array: The predicted probabilities for each class.
        """
        return self.model.predict(X)
