# Define the class for processing text data
import pandas as pd
class TextDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.emotion_columns = [
            'Anger yes / no', 'Anticipation yes / no', 'Disgust yes / no',
            'Fear yes / no', 'Joy yes / no', 'Sadness yes / no',
            'Surprise yes / no', 'Trust yes / no'
        ]

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df = self.df[self.df['Text'].notna()]
        self.df.replace('---', pd.NA, inplace=True)

    def preprocess_text(self):
        self.df['Text'] = self.df['Text'].str.lower()
        self.df['Text'] = self.df['Text'].str.replace(r'http\S+', '', regex=True)
        self.df['Text'] = self.df['Text'].str.replace(r'@\w+', '', regex=True)
        self.df['Text'] = self.df['Text'].str.replace(r'#\w+', '', regex=True)
        self.df['Text'] = self.df['Text'].str.replace(r'[^\w\s]', '', regex=True)
        self.df['Text'] = self.df['Text'].str.replace(r'\s+', ' ', regex=True)
        self.df['Text'] = self.df['Text'].str.strip()

    def convert_emotions_to_binary(self):
        for column in self.emotion_columns:
            self.df[column] = self.df[column].notna().astype(int)
            
    def get_data(self):
        tfidf_df = self.apply_tfidf()
        # Concatenate the TF-IDF features with the binary emotion labels
        return pd.concat([tfidf_df, self.df[self.emotion_columns].reset_index(drop=True)], axis=1)