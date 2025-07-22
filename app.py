# 📦 Installer (dans requirements.txt pour Streamlit Cloud)
# transformers
torch
pandas
streamlit

# 📚 Imports
import streamlit as st
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ⚙️ Initialisation
st.set_page_config(page_title="Analyse NLP Marketing", layout="wide")
st.title("🧠 Analyse NLP Marketing avec BERT")

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# 🧠 Fonctions NLP
def bert_sentiment(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    score = torch.softmax(output.logits, dim=1).cpu().numpy()[0]
    stars = score.argmax() + 1
    return stars, score

def get_sentiment_label(stars):
    if stars <= 2: return "Négatif"
    elif stars == 3: return "Neutre"
    return "Positif"

def detect_emotion(text):
    joy = ["aime", "j'adore", "super", "heureux", "cool"]
    anger = ["déteste", "nul", "énerve", "colère"]
    fear = ["peur", "angoisse", "risque"]
    txt = text.lower()
    if any(w in txt for w in joy): return "Joie"
    elif any(w in txt for w in anger): return "Colère"
    elif any(w in txt for w in fear): return "Peur"
    return "Neutre"

def detect_intention(text):
    txt = text.lower()
    if any(w in txt for w in ["acheter", "commande", "achat"]): return "Achat"
    if any(w in txt for w in ["problème", "bug", "plainte"]): return "Réclamation"
    if any(w in txt for w in ["propose", "suggestion", "améliorer"]): return "Suggestion"
    if any(w in txt for w in ["info", "comment", "quand"]): return "Demande d'information"
    if any(w in txt for w in ["merci", "parfait", "satisfait"]): return "Satisfaction"
    return "Autre"

def detect_marques(text):
    marques = ["pepsi", "infinix", "nike", "samsung", "orange", "amazon", "dior", "iphone", "carrefour", "huawei",
               "jumia", "apple", "netflix", "spotify", "tecno", "adidas", "air france", "aliexpress", "bmw", "bolt",
               "bouygues", "cdiscount", "coca-cola", "danone", "etsy", "free", "h&m", "hp", "kfc", "microsoft",
               "nestlé", "peugeot", "ryanair", "sfr", "sncf", "sony", "starbucks", "tesla", "toyota", "uber", "uniqlo",
               "verizon", "volkswagen", "zara", "ebay"]
    return ", ".join([m.capitalize() for m in marques if m in text.lower()]) or "Aucune"

def generer_recommandation(sentiment, emotion):
    if sentiment == "Positif":
        return "👍 Continuer ainsi"
    if sentiment == "Négatif":
        if emotion == "Colère": return "⚠️ Éviter les contenus irritants"
        if emotion == "Peur": return "🔍 Rassurer les clients"
        return "🛠️ Améliorer l'approche"
    return "🤔 Ajouter plus d’émotion"

def resumator(text):
    return text if len(text.split()) <= 5 else " ".join(text.split()[:8]) + "..."

# 📤 Analyse
uploaded_file = st.file_uploader("📂 Charger un fichier CSV avec une colonne 'message'", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "message" not in df.columns:
        st.error("❌ Le fichier doit contenir une colonne 'message'.")
    else:
        sentiments, stars_list, emotions, intentions, marques, recommandations, resumes, hashtags = [], [], [], [], [], [], [], []

        for msg in df["message"]:
            stars, _ = bert_sentiment(msg)
            sentiment = get_sentiment_label(stars)
            emotion = detect_emotion(msg)
            intention = detect_intention(msg)
            marque = detect_marques(msg)
            reco = generer_recommandation(sentiment, emotion)
            resume = resumator(msg)
            hash_list = re.findall(r"#\w+", msg)

            sentiments.append(sentiment)
            stars_list.append(stars)
            emotions.append(emotion)
            intentions.append(intention)
            marques.append(marque)
            recommandations.append(reco)
            resumes.append(resume)
            hashtags.append(", ".join(hash_list))

        df["sentiment"] = sentiments
        df["emotion"] = emotions
        df["score_emotion"] = [s * 20 for s in stars_list]
        df["intention"] = intentions
        df["marque"] = marques
        df["recommandation"] = recommandations
        df["resume"] = resumes
        df["hashtags"] = hashtags

        st.success("✅ Analyse terminée !")
        st.dataframe(df)

        # 📊 Visualisations
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plt.figure(figsize=(5,3))
            sns.countplot(data=df, x="sentiment", palette="cool")
            plt.title("Répartition des sentiments")
            st.pyplot(fig1)

        with col2:
            fig2 = plt.figure(figsize=(5,3))
            sns.countplot(data=df, x="emotion", palette="Set2")
            plt.title("Répartition des émotions")
            st.pyplot(fig2)

        fig3 = plt.figure(figsize=(10,3))
        wc = WordCloud(width=800, height=300, background_color="white", stopwords=STOPWORDS).generate(" ".join(df["message"]))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Nuage de mots")
        st.pyplot(fig3)

        # 📥 Export CSV
        st.download_button("⬇️ Télécharger les résultats", df.to_csv(index=False).encode("utf-8"), file_name="resultats_nlp.csv", mime="text/csv")
