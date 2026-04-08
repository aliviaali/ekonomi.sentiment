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
# LOAD MODELS - PERBAIKAN DI SINI
# ============================================================================
@st.cache_resource
def load_models():
    try:
        # Perbaikan: Menggunakan Dictionary {key: value} bukan Set
        model_paths = {
            'nb_baseline': 'nb_baseline.pkl',
            'nb_optimized': 'nb_optimized.pkl',
            'svm_baseline': 'svm_baseline.pkl',
            'svm_optimized': 'svm_optimized.pkl',
            'tfidf': 'tfidf (1).pkl',
            'tools': 'preprocessing_tools.pkl'
        }
        
        # Cek keberadaan file
        for name, path in model_paths.items():
            if not os.path.exists(path):
                st.error(f"❌ File tidak ditemukan di root: {path}")
                return None

        # Load Model Utama
        loaded_models = {
            'nb_baseline': joblib.load(model_paths['nb_baseline']),
            'nb_optimized': joblib.load(model_paths['nb_optimized']),
            'svm_baseline': joblib.load(model_paths['svm_baseline']),
            'svm_optimized': joblib.load(model_paths['svm_optimized']),
            'tfidf': joblib.load(model_paths['tfidf']),
        }
        
        # Load Preprocessing Tools
        tools = joblib.load(model_paths['tools'])
        loaded_models['stemmer'] = tools['stemmer']
        loaded_models['stopword_remover'] = tools['stopword_remover']
        loaded_models['additional_stopwords'] = tools.get('additional_stopwords', [])
        
        return loaded_models
    except Exception as e:
        st.error(f"❌ Error mendalam saat memuat model: {str(e)}")
        return None

# Inisialisasi model
models = load_models()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
def get_preprocessing_steps(text, stemmer, stopword_remover, additional_stopwords):
    steps = {}
    text_lower = text.lower()
    text_clean = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|\d+', '', text_lower)
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    text_clean = ' '.join(text_clean.split())
    
    tokens = text_clean.split()
    filtered = [w for w in tokens if w not in additional_stopwords and len(w) >= 3]
    text_no_stop = stopword_remover.remove(' '.join(filtered))
    
    words_stemmed = [stemmer.stem(w) for w in text_no_stop.split()]
    final_result = ' '.join(words_stemmed)
    
    steps['final'] = {'text': final_result}
    return steps

# ============================================================================
# UI TAMPILAN
# ============================================================================
st.title("📊 Analisis Sentimen Berita Ekonomi")
st.markdown("---")

if models:
    input_text = st.text_area("Masukkan Judul Berita:", "IHSG Menguat Tajam Hari Ini")
    
    if st.button("Analisis Sekarang"):
        steps = get_preprocessing_steps(
            input_text, 
            models['stemmer'], 
            models['stopword_remover'], 
            models['additional_stopwords']
        )
        
        clean_text_input = steps['final']['text']

        if not clean_text_input.strip():
            st.warning("⚠️ Teks hasil preprocessing kosong.")
        else:
            try:
                # Transformasi & Prediksi
                vec = models['tfidf'].transform([clean_text_input])
                model_to_use = models['svm_optimized']
                
                prediction = model_to_use.predict(vec)[0]
                prob = model_to_use.predict_proba(vec)[0]
                confidence = max(prob) * 100
                
                st.subheader("Hasil Preprocessing:")
                st.info(clean_text_input)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentimen", prediction.upper())
                with col2:
                    st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

            except sklearn.exceptions.NotFittedError:
                st.error("❌ Model TF-IDF belum dilatih (Not Fitted).")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
else:
    st.warning("⚠️ Aplikasi belum siap karena model gagal dimuat.")
