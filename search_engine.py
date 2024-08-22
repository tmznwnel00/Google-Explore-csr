import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# Load data
df = pd.read_csv('feature_vectors.csv')
words_list = df.columns[1:].tolist()

node_embeddings_array = np.load('final_embeddings.npy')
node_ids = node_embeddings_array[:, 0]
paper_embeddings = np.load('feature_embedding.npy')

# Normalize paper embeddings
paper_embeddings = normalize(paper_embeddings)

def search(query):
    query_words = set(query.lower().split())
    query_embedding = np.zeros(len(words_list), dtype=np.float32)
    for idx, word in enumerate(words_list):
        if word in query_words:
            query_embedding[idx] = 1.0

    # Normalize query embedding
    query_embedding = query_embedding.reshape(1, -1)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_embedding, paper_embeddings)

    # Get top k results
    k = 10

    top_k_indices = np.argsort(similarity_scores[0])[-k:][::-1]
    top_k_scores = similarity_scores[0][top_k_indices]

    return top_k_indices