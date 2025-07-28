from fasttext import load_model
import fasttext.util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import kagglehub
from rapidfuzz import fuzz
import re

def fuzzy_match(query, candidate):
    # Calculate fuzzy match score
    return fuzz.token_set_ratio(query.lower(), candidate.lower())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    return text.strip()
# Download latest version
#path = kagglehub.dataset_download("murtadhayaseen/ccar300bin")

#print("Path to dataset files:", path)

# Load cleaned dataset
df = pd.read_csv("cleaned_capgemini_locations.csv")

# Combine fields into one searchable string
df['full_address'] = df['company_name'] + ", " + df['address'] + ", " + df['city']
addresses = df['full_address'].tolist()

#-------------------------------------------------




# Load FastText English vectors
# Downloads cc.en.300.bin if not present
fasttext.util.download_model('en', if_exists='ignore')
model = load_model('cc.en.300.bin')
#fasttext.util.reduce_model(model, 100)
print("MOdel loaded----")
# Function to get sentence vector using FastText
def sentence_vector(text, model):
    return model.get_sentence_vector(text.lower())

# Embed all addresses
addresses_cleaned = [clean_text(a) for a in addresses]
address_vectors = np.array([sentence_vector(a, model) for a in addresses_cleaned])
np.save("capgemini_address_vectors.npy", address_vectors)
address_vectors = np.load("capgemini_address_vectors.npy")

print("Embedding Done----")

# Query input
query = "jss stp park bhubaneswar"
query_vector = sentence_vector(query, model).reshape(1, -1)

# Cosine similarity
scores = cosine_similarity(query_vector, address_vectors).flatten()
top_k = 5
top_indices = scores.argsort()[::-1][:top_k]
alpha = 0.7
beta = 0.3

result_dict = {}

for i , addr in enumerate(addresses):
    glove_score = scores[i]
    fuzzy_score = fuzzy_match(query, addr)
    combined_score = alpha * glove_score + beta * fuzzy_score / 100

    result_dict[f"match_{i+1}"] = {
        "address": addr,
        "cosine_score": round(glove_score,3),
        "fuzzy_score": round(fuzzy_score,3),
        "combined_score": round(combined_score,3)
    }

sorted_result = sorted(result_dict.items(),key=lambda x: x[1]['combined_score'], reverse=True)

#print(sorted_result[:top_k])

for i ,(k,info) in enumerate(sorted_result[:top_k]):
    print(f"✅ {info['address']} — Cosine Score: {info['cosine_score']:.4f}, Fuzzy Score: {info['fuzzy_score']:.4f}, Combined Score: {info['combined_score']:.4f}")

