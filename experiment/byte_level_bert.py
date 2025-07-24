from transformers import AutoTokenizer, T5EncoderModel
# ByT5 small is compact and fast

import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = T5EncoderModel.from_pretrained("google/byt5-small")
model.eval()


# Step 1: Load address database
df = pd.read_csv("data\company_address.csv")  # must contain 'full_address' column
df["full_address"] =  df["city"]+ " " + df["address"] #df["company_name"] + " " +
addresses = df["full_address"].astype(str).tolist()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).numpy()

query = "jss park bhubaneswar"
address_vectors = np.vstack([get_embedding(a) for a in addresses])
query_vector = get_embedding(query)

scores = cosine_similarity(query_vector, address_vectors).flatten()
top_indices = scores.argsort()[::-1][:5]

print("ByT5 (Byte-level Transformer) Results:")
for idx in top_indices:
    print(f"✅ {addresses[idx]} — Score: {scores[idx]:.4f}")
