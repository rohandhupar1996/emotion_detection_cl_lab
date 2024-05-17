from sklearn.decomposition import PCA
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data loading and TF-IDF application (assuming your data is loaded into 'df')
text_data = ['This is the first document.',
             'This document is the second document.',
             'And this is the third one.',
             'Is this the first document?']

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_data)

# Applying PCA
pca = PCA(n_components=2)  # reduce dimensions to 2 for simplicity
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

# Creating a DataFrame for the PCA results
pca_df = pd.DataFrame(tfidf_pca, columns=['PC1', 'PC2'])
print(pca_df)
#################################
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample data
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Tokenization of the documents
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Training the Word2Vec model
model = Word2Vec(tokenized_docs, vector_size=50, window=2, min_count=1, workers=4)

# Getting the vector for a word
word_vector = model.wv['document']  # Example: Get vector for 'document'
print(word_vector)

################
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer (vocabulary) and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
text = "Here is some text to encode"
encoded_input = tokenizer(text, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    output = model(**encoded_input)

# Get the embeddings for the last layer
embeddings = output.last_hidden_state

# You can average the token embeddings to get a single vector for the entire text
sentence_embedding = torch.mean(embeddings, dim=1)
print(sentence_embedding)

