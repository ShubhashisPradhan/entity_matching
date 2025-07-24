import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load address database
df = pd.read_csv("data\company_address.csv")  # must contain 'full_address' column
df["full_address"] = df["company_name"] + " " + df["city"]+ " " + df["address"] 
addresses = df["full_address"].astype(str).tolist()
#print(addresses[0])
# Step 3: Your partial query
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 2), lowercase=True)
query = "capg jas park bbsr"
#query_vector = vectorizer.transform([query])

# Word-level (or token-based) TF-IDF

address_vectors = vectorizer.fit_transform(addresses)
query_vector = vectorizer.transform([query])
scores = cosine_similarity(query_vector, address_vectors).flatten()
top_indices = scores.argsort()[::-1][:5]

print("TF-IDF + Word-level Results:")
for idx in top_indices:
    print(f"✅ {addresses[idx]} — Score: {scores[idx]:.4f}")
