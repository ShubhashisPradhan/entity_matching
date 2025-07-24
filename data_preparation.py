import pandas as pd


list_of_add = []
data = ""
with open("data/address.txt", 'r') as f:
    data = f.read()
    print(type(data))
    list_of_add = data.split("**")


json_full_address = []

for item in list_of_add:
    json_dt = {}
    add = item.split("--")
    # print(add)
    json_dt["company_name"]  = str(add[1]).replace("\n", "").lower() if str(add[1]).__contains__("\n") else str(add[1]).lower()
    json_dt["address"] = str(add[2]).replace("\n", "").lower() if str(add[2]).__contains__("\n") else str(add[2]).lower()
    json_dt["city"] = str(add[0]).replace("\n", "").lower() if str(add[0]).__contains__("\n") else str(add[0]).lower()

    json_full_address.append(json_dt)

data = pd.DataFrame(json_full_address)
data.to_csv("data/company_address.csv", index=False, encoding='utf-8')

#-------------------- cleaning of data --------------------

import pandas as pd
import re

# Load data
df = pd.read_csv("data/company_address.csv")

# Function to clean address
def clean_address(text):
    if pd.isna(text):
        return text
    text = str(text)

    # Fix encoding issues
    text = text.replace("â€“", "-").replace("â", "")

    # Remove unwanted newlines or excessive spaces
    text = re.sub(r'\s+', ' ', text)
    #text = re.sub(r'\s+', ' ', text)

    # Fix misplaced punctuation and remove double commas
    text = re.sub(r',+', ',', text)

    # Standardize common words
    text = re.sub(r'india$', '', text, flags=re.IGNORECASE)
    text = text.replace("ex madras", "").replace("ex calcutta", "")
    text = text.replace("–", "-").replace("â€“", "-").replace("â", "")
    text = text.replace("–", " ")

    return text.strip(" ,")

# Function to clean city
def clean_city(text):
    if pd.isna(text):
        return text
    text = str(text).lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace("–", " ")  # Remove anything in brackets
    return text.strip()


# Apply cleaning
df['address'] = df['address'].apply(clean_address)
df['city'] = df['city'].apply(clean_city)

# Optional: Title case company name
df['company_name'] = df['company_name'].str.strip().str.title()

# Save cleaned data
df.to_csv("cleaned_capgemini_locations.csv", index=False)
