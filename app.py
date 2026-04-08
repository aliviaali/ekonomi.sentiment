import streamlit as st
import joblib
import pandas as pd
import re
import string
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import plotly.graph_objects as go

# ============================================================================
# SOLUSI ERROR: DEFINISI FUNGSI UNTUK PICKLE
# ============================================================================
# WAJIB: Pickle memerlukan fungsi ini ada di level modul utama (main) 
# agar model bisa di-load tanpa error "AttributeError: clean_text"
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
# LOAD MODELS - DISESUAIKAN DENGAN NAMA FILE DI GITHUB ANDA
# ============================================================================
@st.cache_resource
def load_models():
    try:
        # Nama file disesuaikan persis dengan gambar (menggunakan spasi dan angka 1)
        model_paths = {
            'nb_baseline': 'nb_baseline (1).pkl',
            'nb_optimized': 'nb_optimized (1).pkl',
            'svm_baseline': 'svm_baseline (1).pkl',
            'svm_optimized': 'svm_optimized (1).pkl',
            'tfidf': 'tfidf.pkl',
            'tools': 'preprocessing_tools.pkl'
        }
        
        # Validasi keberadaan file sebelum di-load
        for name, path in model_paths.items():
            if not os.path.exists(path):
                st.error(f"❌ File tidak ditemukan di GitHub: {path}")
                return None

        # Proses loading
        models = {
            'nb_baseline': joblib.load(model_paths['nb_baseline']),
            'nb_optimized': joblib.load(model_paths['nb_optimized']),
            'svm_baseline': joblib.load(model_paths['svm_baseline']),
            'svm_optimized': joblib.load(model_paths['svm_optimized']),
            'tfidf': joblib.load(model_paths['tfidf']),
        }
        
        # Load tools Sastrawi & Stopwords
        tools = joblib.load(model_paths['tools'])
        models['stemmer'] = tools['stemmer']
        models['stopword_remover'] = tools['stopword_remover']
        # Gunakan .get() untuk menghindari error jika key tidak ada
        models['additional_stopwords'] = tools.get('additional_stopwords', [])
        
        return models
    except Exception as e:
        st.error(f"❌ Error mendalam saat memuat model: {str(e)}")
        return None

# Panggil fungsi load
models = load_models()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
def get_preprocessing_steps(text, stemmer, stopword_remover, additional_stopwords):
    steps = {}
    # Original
    steps['original'] = {'text': text, 'char': len(text), 'words': len(text.split())}
    
    # Case Folding
    text_lower = text.lower()
    steps['casefolding'] = {'text': text_lower, 'char': len(text_lower), 'words': len(text_lower.split())}
    
    # Cleaning
    text_clean = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+|\d+', '', text_lower)
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    text_clean = ' '.join(text_clean.split())
    steps['cleaning'] = {'text': text_clean, 'char': len(text_clean), 'words': len(text_clean.split())}
    
    # Tokenization
    tokens = text_clean.split()
    steps['tokenization'] = {'tokens': tokens, 'count': len(tokens)}
    
    # Stopword Removal
    # Hapus kata yang kurang dari 3 huruf dan ada di list stopwords
    filtered = [w for w in tokens if w not in additional_stopwords and len(w) >= 3]
    text_no_stop = stopword_remover.remove(' '.join(filtered))
    steps['stopword_removal'] = {'tokens': text_no_stop.split(), 'count': len(text_no_stop.split())}
    
    # Stemming
    words_stemmed = [stemmer.stem(w) for w in text_no_stop.split()]
    steps['stemming'] = {'tokens': words_stemmed, 'count': len(words_stemmed)}
    
    # Final
    final_result = ' '.join(words_stemmed)
    steps['final'] = {'text': final_result, 'char': len(final_result), 'words': len(words_stemmed)}
    
    return steps

# ============================================================================
# UI TAMPILAN (HEADER & INPUT)
# ============================================================================
st.title("📊 Analisis Sentimen Berita Ekonomi")
st.markdown("---")

if models:
    input_text = st.text_area("Masukkan Judul Berita:", "IHSG Menguat Tajam Hari Ini")
    
    if st.button("Analisis Sekarang"):
        # Jalankan Preprocessing
        steps = get_preprocessing_steps(
            input_text, 
            models['stemmer'], 
            models['stopword_remover'], 
            models['additional_stopwords']
        )
        
        # Tampilkan hasil teks yang sudah bersih
        st.subheader("Hasil Preprocessing (Clean Text):")
        st.info(steps['final']['text'])
        
        # Transformasi ke TF-IDF
        vec = models['tfidf'].transform([steps['final']['text']])
        
        # Prediksi (Contoh menggunakan SVM Optimized)
        prediction = models['svm_optimized'].predict(vec)[0]
        prob = models['svm_optimized'].predict_proba(vec)[0]
        confidence = max(prob) * 100
        
        # Tampilkan Hasil
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentimen", prediction.upper())
        with col2:
            st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
else:
    st.warning("⚠️ Aplikasi belum siap karena model gagal dimuat. Periksa log error di atas.")
