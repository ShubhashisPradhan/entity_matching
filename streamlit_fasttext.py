import streamlit as st
import pandas as pd
import numpy as np
from fasttext import load_model
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import re

# ---------------- Load model and data ----------------
@st.cache_resource(show_spinner=True)
def load_fasttext_model():
    fasttext.util.download_model('en', if_exists='ignore')
    model = load_model('cc.en.300.bin')
    #fasttext.util.reduce_model(model, 100)
    return model

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("cleaned_capgemini_locations.csv")
    df['full_address'] = df['company_name'] + ", " + df['address'] + ", " + df['city']
    df['cleaned_address'] = df['full_address'].apply(clean_text)
    address_vectors = np.load("capgemini_address_vectors.npy")
    return df, address_vectors

# ---------------- Helper functions ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentence_vector(text, model):
    return model.get_sentence_vector(text)

def hybrid_score(query_text, candidate_text, query_vec, candidate_vec, alpha=0.7):
    cos_sim = cosine_similarity([query_vec], [candidate_vec])[0][0]
    fuzz_sim = fuzz.token_set_ratio(query_text, candidate_text) / 100.0
    return alpha * cos_sim + (1 - alpha) * fuzz_sim

def match_address(query, df, address_vectors, model, top_k=5):
    cleaned_query = clean_text(query)
    query_vec = sentence_vector(cleaned_query, model)
    
    scores = []
    fuzzy_scr =[]
    cosine_scr = []
    for i in range(len(df)):
        vec = address_vectors[i]
        cand_text = df.iloc[i]['cleaned_address']
        #score = hybrid_score(cleaned_query, cand_text, query_vec, vec)
        cos_sim = cosine_similarity([query_vec], [vec])[0][0]
        fuzz_sim = fuzz.token_set_ratio(cleaned_query, cand_text) / 100.0
        score = 0.7 * cos_sim + (1 - 0.7) * fuzz_sim
        scores.append(score)
        fuzzy_scr.append(fuzz_sim)
        cosine_scr.append(cos_sim)

    df['match_score'] = scores
    df['fuzzy_score'] = fuzzy_scr
    df['cosine_score'] = cosine_scr
    top_matches = df.sort_values(by='match_score', ascending=False).head(top_k)
    return top_matches[['company_name', 'address', 'city', 'fuzzy_score','cosine_score','match_score']]
    #return top_matches[['company_name', 'address', 'city', 'fuzzy_score','match_score','']]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Capgemini Address Matcher", layout="centered")

st.title("🏢 Capgemini Address Matcher")
st.markdown("Enter a **partial address** or keyword to find the most likely Capgemini office match.")

query_input = st.text_input("🔍 Enter partial address or location", "")

model = load_fasttext_model()
df, address_vectors = load_data()

if query_input:
    with st.spinner("Matching..."):
        result_df = match_address(query_input, df, address_vectors, model, top_k=5)

    st.subheader("🔗 Top Matches:")
    st.table(result_df.style.format({"match_score": "{:.4f}"}))
