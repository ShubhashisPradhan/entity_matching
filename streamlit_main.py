import streamlit as st
import subprocess

st.set_page_config(page_title="Model Launcher", layout="centered")
st.title("🔀 Entity Matcher Launcher")

model_file_map = {
    "Google BERT": "streamlit_bert.py",
    "CANINE": "streamlit_canine.py",
    "ByT5": "streamlit_byt5.py",
    "FastText": "streamlit_fasttext.py",
    "GloVe": "streamlit_glove.py",
    "Sentence Transformer": "streamlit_sent_transformer.py",
}

model_choice = st.selectbox("Choose a model to launch:", list(model_file_map.keys()))

if st.button("🚀 Launch Selected Model"):
    selected_file = model_file_map[model_choice]
    st.success(f"Launching: `{selected_file}`...")
    subprocess.Popen(["streamlit", "run", selected_file])
    st.stop()
