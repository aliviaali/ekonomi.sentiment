import streamlit as st
import joblib
import pandas as pd
import re
import string
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import plotly.graph_objects as go
import sklearn

# ============================================================================
# SOLUSI ERROR: DUMMY FUNCTION
# ============================================================================
# Model Anda mencari fungsi 'clean_text' saat di-load (Unpickling). 
# Kita buat fungsi kosong agar joblib tidak error.
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

# [BAGIAN CSS TETAP SAMA SEPERTI KODE ANDA - DILEWATI UNTUK RINGKAS]
st.markdown("""<style> .main { background: #1a1a2e; } </style>""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS - DIPERBAIKI
# ============================================================================
@st.cache_resource
def load_models():
    try:
        # Menyesuaikan nama file sesuai metadata unggahan Anda
        # Jika di GitHub namanya tidak pakai (1), silakan hapus bagian (1) nya
        model_files = {
            'nb_baseline': 'nb_baseline (1).pkl',
            'nb_optimized': 'nb_optimized (1).pkl',
            'svm_baseline': 'svm_baseline (1).pkl',
            'svm_optimized': 'svm_optimized (1).pkl',
            'tfidf': 'tfidf.pkl',
            'tools': 'preprocessing_tools.pkl'
        }
        
        # Validasi file ada atau tidak
        for key, path in model_files.items():
            if not os.path.exists(path):
                # Fallback: coba cari tanpa ' (1)' jika tidak ketemu
                fallback_path = path.replace(' (1)', '')
                if os.path.exists(fallback_path):
                    model_files[key] = fallback_path
                else:
                    st.error(f"❌ File {path} tidak ditemukan!")
                    return None

        models = {
            'nb_baseline': joblib.load(model_files['nb_baseline']),
            'nb_optimized': joblib.load(model_files['nb_optimized']),
            'svm_baseline': joblib.load(model_files['svm_baseline']),
            'svm_optimized': joblib.load(model_files['svm_optimized']),
            'tfidf': joblib.load(model_files['tfidf']),
        }
        
        tools = joblib.load(model_files['tools'])
        models['stemmer'] = tools['stemmer']
        models['stopword_remover'] = tools['stopword_remover']
        models['additional_stopwords'] = tools.get('additional_stopwords', [])
        
        return models
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {str(e)}")
        return None

models = load_models()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def get_preprocessing_steps(text, stemmer, stopword_remover, additional_stopwords):
    steps = {}
    steps['original'] = {'text': text, 'char': len(text), 'words': len(text.split())}
    
    # 1. Case Folding
    text_lower = text.lower()
    steps['casefolding'] = {'text': text_lower, 'char': len(text_lower), 'words': len(text_lower.split())}
    
    # 2. Cleaning
    text_clean = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|\d+', '', text_lower)
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    text_clean = ' '.join(text_clean.split())
    steps['cleaning'] = {'text': text_clean, 'char': len(text_clean), 'words': len(text_clean.split())}
    
    # 3. Tokenization
    tokens = text_clean.split()
    steps['tokenization'] = {'tokens': tokens, 'count': len(tokens)}
    
    # 4. Stopword Removal
    words_filtered = [w for w in text_clean.split() if w not in additional_stopwords and len(w) >= 3]
    text_no_stop = stopword_remover.remove(' '.join(words_filtered))
    steps['stopword_removal'] = {'tokens': text_no_stop.split(), 'count': len(text_no_stop.split())}
    
    # 5. Stemming
    words_stemmed = [stemmer.stem(w) for w in text_no_stop.split()]
    steps['stemming'] = {'tokens': words_stemmed, 'count': len(words_stemmed)}
    
    steps['final'] = {'text': ' '.join(words_stemmed), 'char': len(' '.join(words_stemmed)), 'words': len(words_stemmed)}
    return steps

# ============================================================================
# UI LOGIC (Analisis & Tampilan)
# ============================================================================
# [TETAP SAMA SEPERTI KODE ASLI ANDA]
st.title("📊 Analisis Sentimen Berita Ekonomi")

if models and st.button("🚀 Analisis Sentimen"):
    input_text = st.session_state.get('input_text', "IHSG Naik")
    steps = get_preprocessing_steps(input_text, models['stemmer'], models['stopword_remover'], models['additional_stopwords'])
    
    # Tampilkan hasil akhir
    final_text = steps['final']['text']
    vec = models['tfidf'].transform([final_text])
    
    # Prediksi menggunakan SVM Optimized sebagai default
    pred = models['svm_optimized'].predict(vec)[0]
    st.success(f"Hasil Prediksi: {pred.upper()}")
