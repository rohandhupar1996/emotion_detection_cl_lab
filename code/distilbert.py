import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling1D
from tensorflow.keras.models import Model

class DistilBERTModel:
    def __init__(self, max_len, num_classes):
        """
        Initializes the DistilBERTModel with the specified maximum sequence length and number of output classes.
        It loads the DistilBERT tokenizer and DistilBERT base model from the transformers library, and builds the final model.

        Parameters:
        max_len (int): The maximum length of input sequences.
        num_classes (int): The number of output classes for classification.
        """
        self.max_len = max_len
        self.num_classes = num_classes
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.base_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the DistilBERT-based model with additional pooling and dense layers.

        Returns:
        Model: A compiled Keras model ready for training.
        """
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
        """
        Trains the model on the provided texts and labels.

        Parameters:
        texts (list of str): The input text data for training.
        labels (np.array): The one-hot encoded labels for the training data.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size for training.
        validation_split (float, optional): The proportion of training data to use for validation. Default is 0.1.

        Returns:
        History: A Keras History object containing training history.
        """
        encodings = self.tokenizer(texts, max_length=self.max_len, padding=True, truncation=True, return_tensors="tf")
        return self.model.fit(
            [encodings['input_ids'], encodings['attention_mask']],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def evaluate(self, texts, labels):
        """
        Evaluates the model on the provided texts and labels.

        Parameters:
        texts (list of str): The input text data for evaluation.
        labels (np.array): The one-hot encoded labels for the evaluation data.

        Returns:
        tuple: The loss and accuracy of the model on the evaluation data.
        """
        encodings = self.tokenizer(texts, max_length=self.max_len, padding=True, truncation=True, return_tensors="tf")
        return self.model.evaluate([encodings['input_ids'], encodings['attention_mask']], labels)

    def predict(self, texts):
        """
        Predicts the labels for the provided texts.

        Parameters:
        texts (list of str): The input text data for prediction.

        Returns:
        np.array: The predicted probabilities for each class.
        """
        encodings = self.tokenizer(texts, max_length=self.max_len, padding=True, truncation=True, return_tensors="tf")
        return self.model.predict([encodings['input_ids'], encodings['attention_mask']])
