from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fuzzywuzzy import fuzz 


#glove_input = "glove.6B\glove.6B.200d.txt"
#glove_model = KeyedVectors.load_word2vec_format(glove_input, binary=False, no_header=True)
#glove_model.save("glove.6B.300d.gensim")
glove_model = KeyedVectors.load("glove.6B.300d.gensim")
def sentence_vector(text, model):
    words = text.lower().split()
    valid_vectors = [glove_model[word] for word in words if word in model]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)


def fuzzy_match(query, address):
    # Calculate fuzzy match score
    return fuzz.token_set_ratio(query.lower(), address.lower())

# Your database
df = pd.read_csv("cleaned_capgemini_locations.csv")
df["full_address"] =  df["company_name"] + " " + df["city"]+ " " + df["address"] 
addresses = df["full_address"].astype(str).tolist()


# Embed all
address_vectors = np.array([sentence_vector(a, glove_model) for a in addresses])

# Embed the query
query = " campus 5b , marrathali"
query_vector = sentence_vector(query, glove_model).reshape(1, -1)
#print(query_vector)
# Cosine similarity
scores = cosine_similarity(query_vector, address_vectors).flatten()
top_k = 5
top_indices = scores.argsort()[::-1][:top_k]
#print(top_indices)
# Display
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





