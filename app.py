import streamlit as st
import joblib
import pandas as pd
import re
import string
import os
import sklearn
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import plotly.graph_objects as go

# ============================================================================
# SOLUSI ERROR: DEFINISI FUNGSI UNTUK PICKLE
# ============================================================================
def clean_text(text):
    return text

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Analisis Sentimen Berita Ekonomi",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# LOAD MODELS - SESUAI NAMA FILE DI GITHUB (DENGAN ANGKA 1)
# ============================================================================
@st.cache_resource
def load_models():
    try:
        model_paths = {
            'nb_baseline': 'nb_baseline.pkl',
            'nb_optimized': 'nb_optimized.pkl',
            'svm_baseline': 'svm_baseline.pkl',
            'svm_optimized': 'svm_optimized.pkl',
            'tfidf': 'tfidf.pkl',
            'tools': 'preprocessing_tools.pkl'
        }
        
        for name, path in model_paths.items():
            if not os.path.exists(path):
                st.error(f"❌ File tidak ditemukan: {path}")
                return None

        models = {
            'nb_baseline': joblib.load(model_paths['nb_baseline']),
            'nb_optimized': joblib.load(model_paths['nb_optimized']),
            'svm_baseline': joblib.load(model_paths['svm_baseline']),
            'svm_optimized': joblib.load(model_paths['svm_optimized']),
            'tfidf': joblib.load(model_paths['tfidf']),
        }
        
        tools = joblib.load(model_paths['tools'])
        models['stemmer'] = tools['stemmer']
        models['stopword_remover'] = tools['stopword_remover']
        models['additional_stopwords'] = tools.get('additional_stopwords', [])
        
        return models
    except Exception as e:
        st.error(f"❌ Error mendalam saat memuat model: {str(e)}")
        return None

models = load_models()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
def get_preprocessing_steps(text, stemmer, stopword_remover, additional_stopwords):
    steps = {}
    steps['original'] = {'text': text, 'char': len(text), 'words': len(text.split())}
    
    text_lower = text.lower()
    steps['casefolding'] = {'text': text_lower, 'char': len(text_lower), 'words': len(text_lower.split())}
    
    text_clean = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|\d+', '', text_lower)
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    text_clean = ' '.join(text_clean.split())
    steps['cleaning'] = {'text': text_clean, 'char': len(text_clean), 'words': len(text_clean.split())}
    
    tokens = text_clean.split()
    steps['tokenization'] = {'tokens': tokens, 'count': len(tokens)}
    
    filtered = [w for w in tokens if w not in additional_stopwords and len(w) >= 3]
    text_no_stop = stopword_remover.remove(' '.join(filtered))
    steps['stopword_removal'] = {'tokens': text_no_stop.split(), 'count': len(text_no_stop.split())}
    
    words_stemmed = [stemmer.stem(w) for w in text_no_stop.split()]
    steps['stemming'] = {'tokens': words_stemmed, 'count': len(words_stemmed)}
    
    final_result = ' '.join(words_stemmed)
    steps['final'] = {'text': final_result, 'char': len(final_result), 'words': len(words_stemmed)}
    
    return steps

# ============================================================================
# UI TAMPILAN
# ============================================================================
st.title("📊 Analisis Sentimen Berita Ekonomi")
st.markdown("---")

if models:
    input_text = st.text_area("Masukkan Judul Berita:", "IHSG Menguat Tajam Hari Ini")
    
    if st.button("Analisis Sekarang"):
        # 1. Jalankan Preprocessing
        steps = get_preprocessing_steps(
            input_text, 
            models['stemmer'], 
            models['stopword_remover'], 
            models['additional_stopwords']
        )
        
        clean_text_input = steps['final']['text']

        # 2. Validasi Teks Hasil Preprocessing
        if not clean_text_input.strip():
            st.warning("⚠️ Teks hasil preprocessing kosong. Mohon masukkan judul berita yang lebih lengkap.")
        else:
            try:
                # 3. Transformasi ke TF-IDF
                vec = models['tfidf'].transform([clean_text_input])
                
                # 4. Prediksi (Menggunakan SVM Optimized)
                model_to_use = models['svm_optimized']
                prediction = model_to_use.predict(vec)[0]
                prob = model_to_use.predict_proba(vec)[0]
                confidence = max(prob) * 100
                
                # 5. Tampilkan Hasil
                st.subheader("Hasil Preprocessing (Clean Text):")
                st.info(clean_text_input)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentimen", prediction.upper())
                with col2:
                    st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

            except sklearn.exceptions.NotFittedError:
                st.error("❌ Model TF-IDF belum dilatih (Not Fitted).")
                st.info("Saran: Pastikan di Google Colab Anda menjalankan `tfidf.fit(X_train)` sebelum melakukan `joblib.dump`.")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
else:
    st.warning("⚠️ Aplikasi belum siap karena model gagal dimuat. Periksa file .pkl di GitHub Anda.")
