import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from fuzzywuzzy import fuzz
import torch
import re
from rapidfuzz import fuzz, distance

# -------------------------
# Helper Functions
# -------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.numpy()

def fuzzy_score(query, address):
    return fuzz.token_set_ratio(query.lower(), address.lower()) / 100.0

def levenshtein_similarity(query, address):
    return 1 - distance.Levenshtein.normalized_distance(clean_text(query), clean_text(address))

def jaccard_similarity(s1, s2):
    set1, set2 = set(clean_text(s1)), set(clean_text(s2))
    return len(set1 & set2) / len(set1 | set2)

# -------------------------
# Load BERT
# -------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# -------------------------
# Load and Prepare Dataset
# -------------------------
df = pd.read_csv("cleaned_capgemini_locations.csv")
df["full_address"] = df["company_name"].fillna('') + " " + df["city"].fillna('') + " " + df["address"].fillna('')
addresses = df["full_address"].astype(str).tolist()

# -------------------------
# Generate Address Embeddings
# -------------------------
address_vectors = np.array([get_bert_embedding(addr, tokenizer, model) for addr in addresses])

# -------------------------
# Query to Match
# -------------------------
query = "campus 5b , marrathali"
query_vector = get_bert_embedding(query, tokenizer, model).reshape(1, -1)

# Cosine similarity
cosine_scores = cosine_similarity(query_vector, address_vectors).flatten()

# -------------------------
# Scoring Weights
# -------------------------
w_cosine = 0.4
w_fuzzy = 0.3
w_levenshtein = 0.2
w_jaccard = 0.1

# -------------------------
# Scoring and Ranking
# -------------------------
result_dict = {}
for i, addr in enumerate(addresses):
    cos_sim = cosine_scores[i]
    fuzzy = fuzzy_score(query, addr)
    lev_sim = levenshtein_similarity(query, addr)
    jacc = jaccard_similarity(query, addr)

    combined_score = (
        w_cosine * cos_sim +
        w_fuzzy * fuzzy +
        w_levenshtein * lev_sim +
        w_jaccard * jacc
    )

    result_dict[f"match_{i+1}"] = {
        "address": addr,
        "cosine_score": round(cos_sim, 3),
        "fuzzy_score": round(fuzzy, 3),
        "levenshtein_score": round(lev_sim, 3),
        "jaccard_score": round(jacc, 3),
        "combined_score": round(combined_score, 3)
    }

# Sort and show top results
sorted_result = sorted(result_dict.items(), key=lambda x: x[1]['combined_score'], reverse=True)
top_k = 5

for i, (k, info) in enumerate(sorted_result[:top_k]):
    print(f"✅ {info['address']}")
    print(f"   Cosine: {info['cosine_score']:.3f}, Fuzzy: {info['fuzzy_score']:.3f}, "
          f"Levenshtein: {info['levenshtein_score']:.3f}, Jaccard: {info['jaccard_score']:.3f}, "
          f"Combined: {info['combined_score']:.3f}\n")
