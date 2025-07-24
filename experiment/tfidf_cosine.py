import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load address database
df = pd.read_csv("data\company_address.csv")  # must contain 'full_address' column
df["full_address"] = df["company_name"] + " " + df["city"]+ " " + df["address"] 
addresses = df["full_address"].astype(str).tolist()
print(addresses)

# Step 2: TF-IDF vectorizer on character n-grams
#vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
address_vectors = vectorizer.fit_transform(addresses)

# Step 3: Your partial query
query = "capg jas park bbsr"
query_vector = vectorizer.transform([query])
#print(f"🔍 Query: '{query_vector}'\n")

# Step 4: Cosine similarity
cosine_scores = cosine_similarity(query_vector, address_vectors).flatten()

# Step 5: Get top matches
top_k = 5
top_indices = cosine_scores.argsort()[::-1][:top_k]

print(f"\n🔍 Query: '{query}'\nTop Matches:")
for idx in top_indices:
    print(f"✅ {addresses[idx]} — Score: {cosine_scores[idx]:.4f}")
