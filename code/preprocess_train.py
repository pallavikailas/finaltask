import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.impute import SimpleImputer
import numpy as np

file_path = r"D:\final_task-7\dataset\train.csv"

df = pd.read_csv(file_path)

df['text'] = df['text'].str.lower()

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F004-\U0001F0CF"  # Miscellaneous Symbols and Pictographs
        u"\U0001F10D-\U0001F10F"  # Musical Symbols
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['text'] = df['text'].apply(remove_emojis)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['text'] = df['text'].apply(remove_stopwords)


stemmer = SnowballStemmer('english')

def stem_text(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

df['text'] = df['text'].apply(stem_text)

columns_to_fill = ['is_humor', 'humor_controversy','humor_rating','offense_rating',] 

# Fill NaN values with random 0s and 1s
for column in columns_to_fill:
    # Generate random 0s and 1s to fill NaN values
    random_values = np.random.randint(0, 2, size=df[column].isnull().sum())
    
    # Replace NaN values with random 0s and 1s
    df[column][df[column].isnull()] = random_values

# Save the preprocessed DataFrame to a CSV file
df.to_csv('preprocessed_train_2.csv', index=False)

# Check if there are any remaining NaN values
print(df[columns_to_fill].isnull().sum())
