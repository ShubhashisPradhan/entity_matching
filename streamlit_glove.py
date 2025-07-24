import streamlit as st
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fuzzywuzzy import fuzz
import re

# ------------------- Clean Function -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------- Sentence Vector -------------------
def sentence_vector(text, model):
    words = text.lower().split()
    valid_vectors = [model[word] for word in words if word in model]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

# ------------------- Fuzzy Matching -------------------
def fuzzy_match(query, address):
    return fuzz.token_set_ratio(query.lower(), address.lower())

# ------------------- Matching Logic -------------------
def hybrid_match(query, df, vectors, model, alpha=0.7, beta=0.3, top_k=5):
    query_clean = clean_text(query)
    query_vec = sentence_vector(query_clean, model).reshape(1, -1)
    cosine_scores = cosine_similarity(query_vec, vectors).flatten()

    results = []
    for i, addr in enumerate(df["full_address"]):
        glove_score = cosine_scores[i]
        fuzzy_score = fuzzy_match(query_clean, addr)/100
        combined_score = alpha * glove_score + beta * fuzzy_score 

        results.append({
            "address": addr,
            "cosine_score": round(glove_score, 3),
            "fuzzy_score": round(fuzzy_score, 3),
            "combined_score": round(combined_score, 3)
        })

    results = sorted(results, key=lambda x: x["combined_score"], reverse=True)[:top_k]
    return results

# ------------------- Load Data & Model -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_capgemini_locations.csv")
    df["full_address"] = df["company_name"] + " " + df["city"] + " " + df["address"]
    df["full_address"] = df["full_address"].astype(str).apply(clean_text)
    return df

@st.cache_resource
def load_glove_model():
    model = KeyedVectors.load("glove.6B.300d.gensim")  # Assumes pre-converted GloVe format
    return model

@st.cache_data
def precompute_vectors(df, _model):
    return np.array([sentence_vector(addr, _model) for addr in df["full_address"]])


# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="GloVe + Fuzzy Address Matcher", layout="centered")
st.title("📍 Capgemini Address Matcher (GloVe + Fuzzy)")
st.markdown("Enter a **partial or fuzzy address** to find the best matching Capgemini location.")

query = st.text_input("🔍 Enter query (e.g., 'campus 5b marrathali')")

model = load_glove_model()
df = load_data()
vectors = precompute_vectors(df, model)

if query:
    with st.spinner("Matching..."):
        matches = hybrid_match(query, df, vectors, model)

    st.subheader("🔗 Top Matches")
    for match in matches:
        st.markdown(
            f"""
            <div style="padding: 8px; border-bottom: 1px solid #ccc;">
                <b>🏢 Address:</b> {match['address']}<br>
                <b>🧠 Cosine Score:</b> {match['cosine_score']} | 
                <b>🧩 Fuzzy Score:</b> {match['fuzzy_score']} | 
                <b>✅ Combined Score:</b> {match['combined_score']}
            </div>
            """, unsafe_allow_html=True
        )
