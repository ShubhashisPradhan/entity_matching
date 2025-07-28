from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, distance
import numpy as np
import pandas as pd
import re

# ------------------ Load Model ------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ Clean Function ------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------ Embedding Function ------------------
def sentence_vector(text, model):
    return model.encode([clean_text(text)])[0]

# ------------------ Fuzzy Matching ------------------
def fuzzy_match(query, address):
    return fuzz.token_set_ratio(clean_text(query), clean_text(address))

# ------------------ Levenshtein Similarity ------------------
def levenshtein_similarity(query, address):
    return 1 - distance.Levenshtein.normalized_distance(clean_text(query), clean_text(address))

# ------------------ Jaccard Similarity ------------------
def jaccard_similarity(query, address):
    set1 = set(clean_text(query).split())
    set2 = set(clean_text(address).split())
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0

# ------------------ Load Address Data ------------------
df = pd.read_csv("cleaned_capgemini_locations.csv")
df["full_address"] = df["company_name"] + " " + df["city"] + " " + df["address"]
addresses = df["full_address"].astype(str).tolist()

# ------------------ Embed Addresses ------------------
address_vectors = np.array([sentence_vector(a, model) for a in addresses])

# ------------------ Embed Query ------------------
query = "jss stp"
query_vector = sentence_vector(query, model).reshape(1, -1)

# ------------------ Cosine Similarity ------------------
scores = cosine_similarity(query_vector, address_vectors).flatten()

# ------------------ Weight Parameters ------------------
alpha = 0.4   # cosine
beta = 0.3    # fuzzy
gamma = 0.15   # levenshtein
delta = 0.15   # jaccard

result_dict = {}

# ------------------ Combined Score Calculation ------------------
for i, addr in enumerate(addresses):
    cosine_score = scores[i]
    fuzzy_score = fuzzy_match(query, addr) / 100
    levenshtein_score = levenshtein_similarity(query, addr)
    jaccard_score = jaccard_similarity(query, addr)

    combined_score = (
        alpha * cosine_score +
        beta * fuzzy_score +
        gamma * levenshtein_score +
        delta * jaccard_score
    )

    result_dict[f"match_{i+1}"] = {
        "address": addr,
        "cosine_score": round(cosine_score, 3),
        "fuzzy_score": round(fuzzy_score, 3),
        "levenshtein_score": round(levenshtein_score, 3),
        "jaccard_score": round(jaccard_score, 3),
        "combined_score": round(combined_score, 3)
    }

# ------------------ Sort and Display Top-K ------------------
top_k = 5
sorted_result = sorted(result_dict.items(), key=lambda x: x[1]['combined_score'], reverse=True)

for i, (k, info) in enumerate(sorted_result[:top_k]):
    print(f"✅ {info['address']}")
    print(f"   Cosine: {info['cosine_score']:.4f}, Fuzzy: {info['fuzzy_score']:.4f}, "
          f"Levenshtein: {info['levenshtein_score']:.4f}, Jaccard: {info['jaccard_score']:.4f}")
    print(f"   ➤ Combined Score: {info['combined_score']:.4f}\n")
