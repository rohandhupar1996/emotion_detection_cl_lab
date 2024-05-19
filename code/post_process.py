def convert_emotion_df_to_dict(df):
    # Emotion columns (assuming all columns except the last one, 'Text', are emotion labels)
    emotion_columns = df.columns[:-1]
    
    # Convert dataframe to list of dictionaries with binary encoding for presence/absence of emotions
    emotion_dicts = []
    for _, row in df.iterrows():
        emotion_dict = {emotion: 1 if row[emotion] == emotion else 0 for emotion in emotion_columns}
        emotion_dicts.append(emotion_dict)
    
    return emotion_dicts