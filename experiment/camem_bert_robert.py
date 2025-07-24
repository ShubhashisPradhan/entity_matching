from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.eval()

# Step 1: Load address database
df = pd.read_csv("cleaned_capgemini_locations.csv")  # must contain 'full_address' column
df["full_address"] = df["city"]+ " " + df["address"] #df["company_name"] + " " + 
addresses = df["full_address"].astype(str).tolist()
# Function to get mean token embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).numpy()

# Embed addresses
query = "jss park bhubaneswar"
address_vectors = np.vstack([get_embedding(a) for a in addresses])
query_vector = get_embedding(query)

# Similarity
scores = cosine_similarity(query_vector, address_vectors).flatten()
top_indices = scores.argsort()[::-1][:5]

print("RoBERTa Subword Token Embeddings Results:")
for idx in top_indices:
    print(f"✅ {addresses[idx]} — Score: {scores[idx]:.4f}")


# fail