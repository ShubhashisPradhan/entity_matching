from gensim.models import KeyedVectors
import numpy as np

# Load pretrained GoogleNews Word2Vec model (300D)
# (Download from: https://code.google.com/archive/p/word2vec/)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


def sentence_vector(text, model):
    words = text.lower().split()
    valid_vectors = [model[word] for word in words if word in model]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Your database
df = pd.read_csv("cleaned_capgemini_locations.csv")
addresses = df['full_address'].tolist()

# Embed all
address_vectors = np.array([sentence_vector(a, model) for a in addresses])

# Embed the query
query = "capg jas park bhubaneswar"
query_vector = sentence_vector(query, model).reshape(1, -1)

# Cosine similarity
scores = cosine_similarity(query_vector, address_vectors).flatten()
top_k = 5
top_indices = scores.argsort()[::-1][:top_k]

# Display
for idx in top_indices:
    print(f"✅ {addresses[idx]} — Score: {scores[idx]:.4f}")
