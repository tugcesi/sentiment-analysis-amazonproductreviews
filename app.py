import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import re
import subprocess
import sys

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# NLTK / TextBlob resource downloads (silent, first-run only)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_nlp_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("brown", quiet=True)
    nltk.download("wordnet", quiet=True)
    # TextBlob corpora (word tokenizer, lemmatizer vs. icin gerekli)
    subprocess.run(
        [sys.executable, "-m", "textblob.download_corpora"],
        check=False,
        capture_output=True,
    )

download_nlp_resources()

# ---------------------------------------------------------------------------
# Text pre-processing function used by the CountVectorizer.
# The name 'ekkok' MUST stay unchanged: vectorizer.joblib was serialised with
# a hard-coded reference to '__main__.ekkok', so joblib cannot deserialise it
# unless this exact name is present in __main__ at load time.
# ---------------------------------------------------------------------------
_stop_words = set(stopwords.words("english"))

def ekkok(text):
    """Lemmatise words and remove English stop-words (analyzer for CountVectorizer)."""
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in _stop_words]

def preprocess_text(text):
    """Apply same preprocessing as training notebook"""
    text = text.lower()
    text = re.sub(r'[^
\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    return text

# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "Negatif 🔴", 1: "Nötr 🟡", 2: "Pozitif 🟢"}
COLOR_MAP = {0: "#FF4B4B", 1: "#FFA500", 2: "#21C354"}
EMOJI_MAP = {0: "😞", 1: "😐", 2: "😊"}

def predict(text: str, model, vectorizer):
    processed_text = preprocess_text(text)
    X = vectorizer.transform([processed_text])
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return label, proba

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Ürün Yorum Analizi",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - cloud.png removed; emoji + markdown used instead
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🛍️ Amazon Yorum Analizi")
    st.markdown("---")
    st.markdown("## ℹ️ Hakkında")
    st.markdown(
        """
        Bu uygulama **Amazon ürün yorumlarını** makine öğrenmesi kullanarak
        üç kategoriye ayırır:

        - 🔴 **Negatif** (1–2 yıldız)
        - 🟡 **Nötr** (3 yıldız)
        - 🟢 **Pozitif** (4–5 yıldız)

        **Model:** Random Forest  
        **Vektörleştirici:** CountVectorizer  
        **Dil:** İngilizce  
        """
    )
    st.divider()
    st.markdown("### 📊 Nasıl Kullanılır?\n 1. **Tek Yorum** sekmesine geç  \n 2. Bir Amazon yorumu yaz  \n 3. **Analiz Et** düğmesine bas  \n — veya —  \n 1. **Toplu Analiz** sekmesine geç  \n 2. CSV dosyasını yükle  \n 3. Sonuçları incele & indir  ")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🛍️ Amazon Ürün Yorum Analizi</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Yapay zeka destekli duygu analizi — Negatif · Nötr · Pozitif</p>',
    unsafe_allow_html=True,
)

tab_single, tab_batch = st.tabs(["✏️ Tek Yorum Analizi", "📂 Toplu Analiz (CSV)"])

# ---------------------------------------------------------------------------
# Tab 1 - Single review
# ---------------------------------------------------------------------------
with tab_single:
    st.markdown('<p class="section-header">Yorumunuzu girin</p>', unsafe_allow_html=True)

    example_reviews = {
        "Örnek seç…": "",
        "Çok beğendim 😊": "This product is absolutely amazing! Works perfectly and fast delivery. Highly recommended!",
        "Fena değil 😐": "The product is okay, nothing special. Does what it's supposed to do.",
        "Hayal kırıklığı 😞": "Terrible quality, broke after two days. Waste of money. Very disappointed.",
    }
    choice = st.selectbox("Hızlı örnek:", list(example_reviews.keys()))
    prefill = example_reviews[choice]

    user_input = st.text_area(
        "Yorum metni (İngilizce):",
        value=prefill,
        height=140,
        placeholder="Type your Amazon product review here…",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_input.strip():
            st.warning("Lütfen bir yorum girin.")
        else:
            with st.spinner("Analiz ediliyor…"):
                model, vectorizer = load_model()
                label, proba = predict(user_input.strip(), model, vectorizer)

            color = COLOR_MAP[label]
            result_label = LABEL_MAP[label]
            emoji = EMOJI_MAP[label]

            st.markdown(
                f'<div class="result-box" style="background-color: {color}22; border: 2px solid {color}; color: {color};">'
                f'{emoji} Tahmin: <strong>{result_label}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### 📊 Olasılık Dağılımı")
            prob_df = pd.DataFrame(
                {"Sınıf": ["Negatif 🔴", "Nötr 🟡", "Pozitif 🟢"], "Olasılık": proba}
            )

            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.barh(
                prob_df["Sınıf"],
                prob_df["Olasılık"],
                color=[COLOR_MAP[0], COLOR_MAP[1], COLOR_MAP[2]],
                edgecolor="none",
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Olasılık")
            ax.set_title("Sınıf Olasılıkları")
            for bar, val in zip(bars, proba):
                ax.text(
                    min(val + 0.02, 0.95),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}",
                    va="center",
                    fontweight="bold",
                )
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------------------------------------------------------------------------
# Tab 2 - Batch CSV analysis
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown('<p class="section-header">CSV Dosyası Yükle</p>', unsafe_allow_html=True)
    st.markdown(
        """
        CSV dosyanızda en az bir **`reviewText`** sütunu bulunmalıdır.  \n        Başka sütunlar varsa aynen korunur.
        """
    )

    uploaded_file = st.file_uploader("CSV dosyasını sürükle & bırak ya da seç:", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")
            st.stop()

        if "reviewText" not in df.columns:
            st.error("CSV dosyasında `reviewText` sütunu bulunamadı.")
        else:
            df["reviewText"] = df["reviewText"].fillna(""").astype(str)
            st.success(f"✅ {len(df):,} yorum yüklendi.")

            with st.spinner("Toplu analiz yapılıyor…"):
                model, vectorizer = load_model()
                processed_texts = [preprocess_text(text) for text in df["reviewText"].tolist()]
                X = vectorizer.transform(processed_texts)
                labels = model.predict(X)
                proba_matrix = model.predict_proba(X)

            df["Tahmin (Sayısal)"] = labels
            df["Duygu"] = [LABEL_MAP[lbl] for lbl in labels]
            df["Negatif Olasılığı"] = proba_matrix[:, 0].round(3)
            df["Nötr Olasılığı"] = proba_matrix[:, 1].round(3)
            df["Pozitif Olasılığı"] = proba_matrix[:, 2].round(3)

            counts = pd.Series(labels).value_counts().sort_index()
            total = len(labels)
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Negatif", f"{counts.get(0, 0):,}", f"{counts.get(0, 0)/total:.1%}")
            col2.metric("🟡 Nötr", f"{counts.get(1, 0):,}", f"{counts.get(1, 0)/total:.1%}")
            col3.metric("🟢 Pozitif", f"{counts.get(2, 0):,}", f"{counts.get(2, 0)/total:.1%}")

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            labels_pie = ["Negatif", "Nötr", "Pozitif"]
            sizes = [counts.get(i, 0) for i in range(3)]
            colors_pie = [COLOR_MAP[0], COLOR_MAP[1], COLOR_MAP[2]]
            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=labels_pie,
                colors=colors_pie,
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax2.set_title("Duygu Dağılımı", fontweight="bold")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown("#### 📋 Sonuç Tablosu (ilk 50 satır)")
            display_cols = ["reviewText", "Duygu", "Negatif Olasılığı", "Nötr Olasılığı", "Pozitif Olasılığı"]
            existing_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[existing_cols].head(50), use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Sonuçları İndir (CSV)",
                data=csv_out,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
