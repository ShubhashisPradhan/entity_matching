import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, distance
import numpy as np
import pandas as pd
import re

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ Utility Functions ------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentence_vector(text, _model):
    return _model.encode([clean_text(text)])[0]

def fuzzy_match(query, address):
    return fuzz.token_set_ratio(clean_text(query), clean_text(address))

def levenshtein_similarity(query, address):
    return 1 - distance.Levenshtein.normalized_distance(clean_text(query), clean_text(address))

def jaccard_similarity(query, address):
    set1 = set(clean_text(query).split())
    set2 = set(clean_text(address).split())
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_capgemini_locations.csv")
    df["full_address"] = df["company_name"] + " " + df["city"] + " " + df["address"]
    return df

df = load_data()
addresses = df["full_address"].astype(str).tolist()

# ------------------ Precompute Embeddings ------------------
@st.cache_data
def compute_address_vectors(addresses, _model):
    return np.array([sentence_vector(addr, _model) for addr in addresses])

address_vectors = compute_address_vectors(addresses, model)

# ------------------ Streamlit UI ------------------
st.title("🏷️ Entity Matcher -  Addresses")

query = st.text_input("🔍 Enter your query:", "jss stp")

top_k = 5

alpha =  0.4
beta =  0.3
gamma = 0.1
delta =  0.2

# Normalize weights (optional)
total = alpha + beta + gamma + delta
alpha, beta, gamma, delta = [x / total for x in (alpha, beta, gamma, delta)]

if st.button("Find Matches"):
    query_vec = sentence_vector(query, model).reshape(1, -1)
    cosine_scores = cosine_similarity(query_vec, address_vectors).flatten()

    result_dict = {}
    for i, addr in enumerate(addresses):
        cos = cosine_scores[i]
        fuzzy = fuzzy_match(query, addr) / 100
        lev = levenshtein_similarity(query, addr)
        jac = jaccard_similarity(query, addr)

        combined = alpha * cos + beta * fuzzy + gamma * lev + delta * jac

        result_dict[f"match_{i+1}"] = {
            "address": addr,
            "cosine": round(cos, 3),
            "fuzzy": round(fuzzy, 3),
            "levenshtein": round(lev, 3),
            "jaccard": round(jac, 3),
            "score": round(combined, 3)
        }

    sorted_results = sorted(result_dict.items(), key=lambda x: x[1]["score"], reverse=True)

    st.subheader(f"📋 Top {top_k} Matches")
    for i, (_, info) in enumerate(sorted_results[:top_k]):
        st.markdown(f"**{i+1}. {info['address']}**")
        st.write(f"Cosine: {info['cosine']}, Fuzzy: {info['fuzzy']}, "
                 f"Levenshtein: {info['levenshtein']}, Jaccard: {info['jaccard']}")
        st.success(f"✅ Combined Score: {info['score']}")
