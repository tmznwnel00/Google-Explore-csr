from collections import defaultdict
import pandas as pd
import os

import firebase_admin
from firebase_admin import credentials, db
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


cred = credentials.Certificate(os.getcwd() + "/explorecsr-3178b-firebase-adminsdk-7kg05-7ed0ddb01b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://explorecsr-3178b-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

ref = db.reference("papers")
papers = ref.get()
paper_data = {paper_id: paper_info['abstract'] for paper_id, paper_info in papers.items()}

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

word_freq = defaultdict(int)
processed_abstracts = {}
for paper_id, abstract in paper_data.items():
    words = abstract.split()
    processed_words = []
    
    for word in words:
        word = word.lower()  # Convert to lowercase
        word = stemmer.stem(word)  # Apply stemming
        if word not in stop_words and word.isalpha():  # Remove stopwords and non-alphabetic tokens
            processed_words.append(word)
            word_freq[word] += 1  # Count word frequency
    
    processed_abstracts[paper_id] = processed_words

filtered_words = {word for word, freq in word_freq.items() if freq >= 10}

vocabulary = list(filtered_words)
feature_vectors = []

for paper_id, processed_words in processed_abstracts.items():
    feature_vector = [1 if word in processed_words else 0 for word in vocabulary]
    feature_vectors.append([paper_id] + feature_vector)

columns = ['paper_id'] + vocabulary

df = pd.DataFrame(feature_vectors, columns=columns)
df.to_csv('feature_vectors.csv', index=False)