# 📦 Installer
!pip install transformers torch torchvision torchaudio wordcloud matplotlib seaborn ipywidgets fpdf deep-translator datasets --quiet

# 📚 Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import ipywidgets as widgets
from IPython.display import display, clear_output
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from fpdf import FPDF
import torch, re, io
from google.colab import files

# ⚙️ Setup modèle
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

# 📊 Affichage
def afficher_resultats(df):
    clear_output()
    print("\n📊 Résultats de l'analyse NLP :")
    display(df[["message", "resume", "sentiment", "emotion", "score_emotion", "marque", "intention", "recommandation"]])

    sns.countplot(data=df, x="intention", palette="Pastel1")
    plt.title("Intentions détectées")
    plt.xticks(rotation=45)
    plt.show()

    sns.countplot(data=df, x="sentiment", palette="cool")
    plt.title("Sentiments")
    plt.show()

    sns.countplot(data=df, x="emotion", palette="Set2")
    plt.title("Émotions")
    plt.show()

    wc = WordCloud(width=800, height=300, background_color="white", stopwords=STOPWORDS).generate(" ".join(df["message"]))
    plt.figure(figsize=(8, 3))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nuage de mots")
    plt.show()

# 📤 Export
def exporter_csv(df):
    df.to_csv("resultats_analyse.csv", index=False)
    files.download("resultats_analyse.csv")

def generer_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport NLP Marketing", ln=True, align='C')
    pdf.ln(10)
    for _, row in df.iterrows():
        pdf.multi_cell(0, 8, f"Message : {row['message']}\nRésumé : {row['resume']}\nSentiment : {row['sentiment']} | Émotion : {row['emotion']} ({row['score_emotion']}/100)\nMarque : {row['marque']}\nIntention : {row['intention']}\nRecommandation : {row['recommandation']}\n---", border=0)
    pdf.output("rapport_marketing.pdf")
    files.download("rapport_marketing.pdf")

# 🔎 Filtres interactifs
def filtrer_resultats(df):
    marques = sorted(df["marque"].unique())
    intentions = sorted(df["intention"].unique())

    filtre_marque = widgets.Dropdown(options=["Toutes"] + marques, description="Marque:")
    filtre_intention = widgets.Dropdown(options=["Toutes"] + intentions, description="Intention:")
    bouton_filtrer = widgets.Button(description="🔎 Filtrer")

    output_filtre = widgets.Output()

    def on_filtrer_clicked(b):
        with output_filtre:
            clear_output()
            df_filtre = df.copy()
            if filtre_marque.value != "Toutes":
                df_filtre = df_filtre[df_filtre["marque"] == filtre_marque.value]
            if filtre_intention.value != "Toutes":
                df_filtre = df_filtre[df_filtre["intention"] == filtre_intention.value]
            if df_filtre.empty:
                print("Aucun résultat pour ce filtre.")
            else:
                afficher_resultats(df_filtre)

    bouton_filtrer.on_click(on_filtrer_clicked)
    display(widgets.HBox([filtre_marque, filtre_intention, bouton_filtrer]), output_filtre)

# 🔁 Fine-tuning
fine_tune_done = False

def fine_tune_et_rafraichir(df_original):
    global model, fine_tune_done
    if fine_tune_done or "label" not in df_original.columns:
        return

    print("🔧 Fine-tuning en cours...")

    dataset = Dataset.from_pandas(df_original[["message", "label"]].dropna())

    def tokenize(batch): return tokenizer(batch["message"], padding=True, truncation=True, max_length=512)
    dataset = dataset.map(tokenize, batched=True)

    num_labels = len(df_original["label"].unique())
    model_ft = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

    training_args = TrainingArguments(
        output_dir="./results_ft", num_train_epochs=3, per_device_train_batch_size=8,
        logging_dir="./logs", save_strategy="epoch", report_to="none"
    )

    trainer = Trainer(model=model_ft, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model("model_finetuned")
    tokenizer.save_pretrained("model_finetuned")

    model = AutoModelForSequenceClassification.from_pretrained("model_finetuned").to(device)
    fine_tune_done = True
    print("✅ Fine-tuning terminé. Ré-analyse avec nouveau modèle...")
    traiter_et_analyser(df_original)

# 🧠 Traitement principal
def traiter_et_analyser(df):
    stars_list, sentiments = [], []
    for msg in df["message"]:
        stars, _ = bert_sentiment(msg)
        stars_list.append(stars)
        sentiments.append(get_sentiment_label(stars))

    df["sentiment"] = sentiments
    df["emotion"] = df["message"].apply(detect_emotion)
    df["score_emotion"] = [s * 20 for s in stars_list]
    df["resume"] = df["message"].apply(resumator)
    df["hashtags"] = df["message"].apply(lambda x: re.findall(r"#\w+", x))
    df["marque"] = df["message"].apply(detect_marques)
    df["intention"] = df["message"].apply(detect_intention)
    df["recommandation"] = df.apply(lambda row: generer_recommandation(row["sentiment"], row["emotion"]), axis=1)

    afficher_resultats(df)
    filtrer_resultats(df)
    display(widgets.Button(description="📥 Exporter CSV", on_click=lambda b: exporter_csv(df)))
    display(widgets.Button(description="📄 Générer PDF", on_click=lambda b: generer_pdf(df)))

    fine_tune_et_rafraichir(df)

# 📂 Upload unique
upload_unique = widgets.FileUpload(accept='.csv', multiple=False, description="📂 Charger le CSV d’analyse")
display(upload_unique)

def handle_upload(change):
    if not upload_unique.value: return
    file = list(upload_unique.value.values())[0]
    df = pd.read_csv(io.BytesIO(file['content']))

    if "message" not in df.columns:
        print("❌ Le fichier doit contenir une colonne 'message'")
        return

    traiter_et_analyser(df)

upload_unique.observe(handle_upload, names='value')

