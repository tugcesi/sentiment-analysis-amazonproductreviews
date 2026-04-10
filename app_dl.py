import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# TensorFlow / Keras
# ---------------------------------------------------------------------------
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Sklearn – TF-IDF (refitted on training data, same as notebook)
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Preprocessing – exactly the same steps as the training notebook
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text

# ---------------------------------------------------------------------------
# Load model + refit TF-IDF on training data (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    # 1) Load Keras model
    model = load_model("sentiment_amazon_model.h5")

    # 2) Derive max_features from model input shape
    max_features = model.input_shape[1]

    # 3) Read training data and apply same label mapping as notebook
    df = pd.read_csv("amazon_reviews.csv")
    df = df.dropna(subset=["reviewText"])
    df = df[["overall", "reviewText"]].copy()

    label_map = {1.0: 0, 2.0: 0, 3.0: 1, 4.0: 2, 5.0: 2}
    df["sentiment"] = df["overall"].map(label_map)
    df = df.dropna(subset=["sentiment"])

    # 4) Preprocess texts
    df["reviewText"] = df["reviewText"].apply(preprocess_text)

    # 5) Refit TF-IDF with same parameters used during training
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf.fit(df["reviewText"])

    return model, tfidf

# ---------------------------------------------------------------------------
# Label / colour / emoji maps (same as app.py)
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "Negatif 🔴", 1: "Nötr 🟡", 2: "Pozitif 🟢"}
COLOR_MAP = {0: "#FF4B4B",   1: "#FFA500",  2: "#21C354"}
EMOJI_MAP = {0: "😞",         1: "😐",        2: "😊"}

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict(text: str, model, tfidf):
    processed = preprocess_text(text)
    X = tfidf.transform([processed]).toarray()
    proba = model.predict(X, verbose=0)[0]
    label = int(np.argmax(proba))
    return label, proba

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Yorum Analizi – Deep Learning",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (same style as app.py)
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
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("cloud.png", use_container_width=True)
    st.markdown("## 🛍️ Hakkında")
    st.markdown(
        """
        Bu uygulama **Amazon ürün yorumlarını** derin öğrenme kullanarak
        üç kategoriye ayırır:

        - 🔴 **Negatif** (1–2 yıldız)
        - 🟡 **Nötr** (3 yıldız)
        - 🟢 **Pozitif** (4–5 yıldız)

        **Model:** Deep Learning (Keras Sequential)  
        **Vektörleştirici:** TF-IDF  
        **Dil:** İngilizce  
        """
    )
    st.divider()
    st.markdown("### 📊 Nasıl Kullanılır?")
    st.markdown(
        """
        1. **Tek Yorum** sekmesine geç  
        2. Bir Amazon yorumu yaz  
        3. **Analiz Et** düğmesine bas  
        — veya —  
        1. **Toplu Analiz** sekmesine geç  
        2. CSV dosyasını yükle  
        3. Sonuçları incele & indir  
        """
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🛍️ Amazon Ürün Yorum Analizi</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Deep Learning destekli duygu analizi — Negatif · Nötr · Pozitif</p>',
    unsafe_allow_html=True,
)

# Load model & vectorizer (shown once at startup)
with st.spinner("Model yükleniyor, lütfen bekleyin…"):
    model, tfidf = load_assets()

tab_single, tab_batch = st.tabs(["✏️ Tek Yorum Analizi", "📂 Toplu Analiz (CSV)"])

# ---------------------------------------------------------------------------
# Tab 1 – Single review
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

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        analyze_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_input.strip():
            st.warning("Lütfen bir yorum girin.")
        else:
            with st.spinner("Analiz ediliyor…"):
                label, proba = predict(user_input.strip(), model, tfidf)

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
# Tab 2 – Batch CSV analysis
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown('<p class="section-header">CSV Dosyası Yükle</p>', unsafe_allow_html=True)
    st.markdown(
        """
        CSV dosyanızda en az bir **`reviewText`** sütunu bulunmalıdır.  
        Başka sütunlar varsa aynen korunur.
        """
    )

    uploaded_file = st.file_uploader("CSV dosyasını sürükle & bırak ya da seç:", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")
            st.stop()

        if "reviewText" not in df_upload.columns:
            st.error("CSV dosyasında `reviewText` sütunu bulunamadı.")
        else:
            df_upload["reviewText"] = df_upload["reviewText"].fillna("").astype(str)
            st.success(f"✅ {len(df_upload):,} yorum yüklendi.")

            with st.spinner("Toplu analiz yapılıyor…"):
                processed_texts = [preprocess_text(t) for t in df_upload["reviewText"].tolist()]
                X_batch = tfidf.transform(processed_texts).toarray()
                proba_matrix = model.predict(X_batch, verbose=0)
                labels = np.argmax(proba_matrix, axis=1)

            df_upload["Duygu"] = [LABEL_MAP[lbl] for lbl in labels]
            df_upload["Negatif Olasılığı"] = proba_matrix[:, 0].round(3)
            df_upload["Nötr Olasılığı"]    = proba_matrix[:, 1].round(3)
            df_upload["Pozitif Olasılığı"] = proba_matrix[:, 2].round(3)

            # Summary metrics
            counts = pd.Series(labels).value_counts().sort_index()
            total = len(labels)
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Negatif", f"{counts.get(0, 0):,}", f"{counts.get(0, 0)/total:.1%}")
            col2.metric("🟡 Nötr",    f"{counts.get(1, 0):,}", f"{counts.get(1, 0)/total:.1%}")
            col3.metric("🟢 Pozitif", f"{counts.get(2, 0):,}", f"{counts.get(2, 0)/total:.1%}")

            # Pie chart
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sizes = [counts.get(i, 0) for i in range(3)]
            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=["Negatif", "Nötr", "Pozitif"],
                colors=[COLOR_MAP[0], COLOR_MAP[1], COLOR_MAP[2]],
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

            # Data table preview
            st.markdown("#### 📋 Sonuç Tablosu (ilk 50 satır)")
            display_cols = ["reviewText", "Duygu", "Negatif Olasılığı", "Nötr Olasılığı", "Pozitif Olasılığı"]
            existing_cols = [c for c in display_cols if c in df_upload.columns]
            st.dataframe(df_upload[existing_cols].head(50), use_container_width=True)

            # Download button
            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Sonuçları İndir (CSV)",
                data=csv_out,
                file_name="sentiment_results_dl.csv",
                mime="text/csv",
            )
