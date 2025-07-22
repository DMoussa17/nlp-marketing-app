
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.title("🧠 Analyse NLP Marketing avec BERT")

uploaded_file = st.file_uploader("📂 Charger un fichier CSV avec une colonne 'message'", type="csv")
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def bert_sentiment(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    scores = torch.softmax(output.logits, dim=1).cpu().numpy()[0]
    stars = scores.argmax() + 1
    return stars, scores

def get_sentiment_label(stars):
    if stars <= 2: return "Négatif"
    elif stars == 3: return "Neutre"
    return "Positif"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "message" not in df.columns:
        st.error("❌ Le fichier doit contenir une colonne 'message'.")
    else:
        sentiments, stars_list = [], []
        for msg in df["message"]:
            stars, _ = bert_sentiment(msg)
            sentiments.append(get_sentiment_label(stars))
            stars_list.append(stars)

        df["sentiment"] = sentiments
        st.success("✅ Analyse terminée !")
        st.dataframe(df[["message", "sentiment"]])
        st.download_button("⬇️ Télécharger résultats", df.to_csv(index=False).encode("utf-8"), file_name="resultats.csv", mime="text/csv")
