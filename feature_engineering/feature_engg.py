from sklearn.feature_extraction.text import TfidfVectorizer

def apply_tfidf(self):
        vectorizer = TfidfVectorizer(max_features=1000)  # Limiting to 1000 features for simplicity
        self.tfidf_matrix = vectorizer.fit_transform(self.df['Text'])
        feature_names = vectorizer.get_feature_names_out()
        # Convert the matrix to a DataFrame to make it more human-readable
        tfidf_df = pd.DataFrame(self.tfidf_matrix.toarray(), columns=feature_names)
        return tfidf_df

    def get_data(self):
        return self.df