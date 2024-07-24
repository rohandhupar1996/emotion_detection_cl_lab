import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertTokenizer

class BERTCNNModel:
    def __init__(self, max_len, num_classes):
        """
        Initializes the BERTCNNModel with the specified maximum sequence length and number of output classes.
        It loads the BERT tokenizer and BERT base model from the transformers library, and builds the final model.

        Parameters:
        max_len (int): The maximum length of input sequences.
        num_classes (int): The number of output classes for classification.
        """
        self.max_len = max_len
        self.num_classes = num_classes
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.base_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the BERT-based model with additional convolutional and dense layers.

        Returns:
        Model: A compiled Keras model ready for training.
        """
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
        encodings = self.tokenizer(texts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="tf")
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
        encodings = self.tokenizer(texts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="tf")
        return self.model.evaluate([encodings['input_ids'], encodings['attention_mask']], labels)

    def predict(self, texts):
        """
        Predicts the labels for the provided texts.

        Parameters:
        texts (list of str): The input text data for prediction.

        Returns:
        np.array: The predicted probabilities for each class.
        """
        encodings = self.tokenizer(texts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="tf")
        return self.model.predict([encodings['input_ids'], encodings['attention_mask']])