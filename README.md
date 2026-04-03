# Amazon Ürün Yorumları Duygu Analizi 🛍️

Makine öğrenmesi (Random Forest) ve Streamlit kullanarak Amazon ürün yorumlarını **Negatif**, **Nötr** ve **Pozitif** olarak sınıflandıran interaktif bir web uygulaması.

## ✨ Özellikler

- 🔍 **Tek Yorum Analizi** – Bir Amazon yorumu girerek anında duygu tahmini alın
- 📊 **Olasılık Grafiği** – Her sınıf için model güven yüzdesi
- 📂 **Toplu CSV Analizi** – CSV dosyası yükleyerek binlerce yorumu aynı anda analiz edin
- 🎨 **Renk Kodlu Sonuçlar** – Negatif 🔴 · Nötr 🟡 · Pozitif 🟢
- ⬇️ **Sonuç İndirme** – Toplu analiz çıktısını CSV olarak indirin

## 🚀 Kurulum

### Gereksinimler

- Python 3.9+

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/tugcesi/sentiment-analysis-amazonproductreviews.git
cd sentiment-analysis-amazonproductreviews

# 2. (İsteğe bağlı) Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Uygulamayı çalıştırın
streamlit run app.py
```

Tarayıcınız otomatik olarak `http://localhost:8501` adresini açar.

## 📁 Proje Yapısı

```
sentiment-analysis-amazonproductreviews/
├── app.py                                      # Streamlit uygulaması
├── requirements.txt                            # Python bağımlılıkları
├── sentiment_model.joblib                      # Eğitilmiş Random Forest modeli
├── vectorizer.joblib                           # CountVectorizer
├── amazon_reviews.csv                          # Veri seti
├── AmazonProductReviewsSentimentAnalysis.ipynb # Model eğitim notebook'u
└── cloud.png                                   # Kelime bulutu görseli
```

## 🤖 Model Detayları

| Özellik | Değer |
|---|---|
| Algoritma | Random Forest Classifier |
| Vektörleştirici | CountVectorizer (1–2 gram) |
| Ön işleme | TextBlob lemmatizasyon + NLTK stop-word eleme |
| Sınıflar | 0 – Negatif · 1 – Nötr · 2 – Pozitif |
| Eğitim verisi | Amazon ürün yorumları |

## 📊 Etiket Dönüşümü

| Yıldız | Duygu |
|---|---|
| ⭐ 1–2 | Negatif |
| ⭐⭐⭐ 3 | Nötr |
| ⭐⭐⭐⭐ 4–5 | Pozitif |

## 📋 CSV Formatı

Toplu analiz için CSV dosyanızda aşağıdaki sütun bulunmalıdır:

```
reviewText
This product is amazing!
Terrible quality, very disappointed.
...
```

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
