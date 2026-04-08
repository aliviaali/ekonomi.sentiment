"""
Aplikasi Web Analisis Sentimen Berita Ekonomi Indonesia
Sesuai dengan BAB IV - Step-by-Step Preprocessing Display
"""

import streamlit as st
import joblib
import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import plotly.graph_objects as go
import sklearn
print(sklearn.__version__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Analisis Sentimen Berita Ekonomi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - SESUAI BAB IV
# ============================================================================
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Title */
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle */
    .subtitle-text {
        font-size: 1.1rem;
        color: #a8dadc;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Step cards - SEPERTI TABEL BAB IV */
    .step-card {
        background: linear-gradient(135deg, rgba(26, 188, 156, 0.1) 0%, rgba(52, 152, 219, 0.1) 100%);
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .step-number {
        display: inline-block;
        background: #3498db;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .step-title {
        color: #3498db;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .step-content {
        color: #ecf0f1;
        font-size: 1rem;
        padding: 0.5rem 0;
        background: rgba(0,0,0,0.2);
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-family: 'Courier New', monospace;
    }
    
    .step-stats {
        color: #95a5a6;
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 0.3rem;
    }
    
    /* Final result box */
    .final-result {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        text-align: center;
    }
    
    .final-result-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .final-result-content {
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    
    /* Prediction box */
    .pred-positive {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.4);
        margin: 1rem 0;
    }
    
    .pred-negative {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(231, 76, 60, 0.4);
        margin: 1rem 0;
    }
    
    .pred-neutral {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(52, 152, 219, 0.4);
        margin: 1rem 0;
    }
    
    /* Table styling */
    .comparison-table {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Token display */
    .token {
        display: inline-block;
        background: rgba(52, 152, 219, 0.3);
        color: #ecf0f1;
        padding: 0.3rem 0.7rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        border: 1px solid rgba(52, 152, 219, 0.5);
    }
    
    /* Stats card */
    .stats-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stats-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3498db;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #95a5a6;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
# Bagian LOAD MODELS yang diperbaiki
@st.cache_resource
def load_models():
    try:
        # Gunakan nama file yang sesuai dengan yang ada di repository
        # Jika di repo namanya ada (1), sesuaikan di sini atau rename filenya
        models = {
            'nb_baseline': joblib.load('nb_baseline.pkl'),
            'nb_optimized': joblib.load('nb_optimized.pkl'),
            'svm_baseline': joblib.load('svm_baseline.pkl'),
            'svm_optimized': joblib.load('svm_optimized.pkl'),
            'tfidf': joblib.load('tfidf.pkl'),
        }
        
        tools = joblib.load('preprocessing_tools.pkl')
        models['stemmer'] = tools['stemmer']
        models['stopword_remover'] = tools['stopword_remover']
        models['additional_stopwords'] = tools['additional_stopwords']
        
        return models
    except FileNotFoundError as e:
        st.error(f"❌ File Model Tidak Ditemukan: {e.filename}")
        st.info("Pastikan nama file di GitHub sama persis dengan yang dipanggil di kode (Case Sensitive).")
        return None
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {str(e)}")
        return None

models = load_models()

# ============================================================================
# PREPROCESSING FUNCTIONS - IDENTIK DENGAN TRAINING!
# ============================================================================

def get_preprocessing_steps(text, stemmer, stopword_remover, additional_stopwords):
    """
    Get preprocessing step-by-step SESUAI TABEL BAB IV
    Returns dict dengan setiap tahap
    """
    steps = {}
    
    # Original
    steps['original'] = {
        'text': text,
        'char': len(text),
        'words': len(text.split())
    }
    
    # Step 1: Case Folding
    text_lower = text.lower()
    steps['casefolding'] = {
        'text': text_lower,
        'char': len(text_lower),
        'words': len(text_lower.split())
    }
    
    # Step 2: Cleaning
    text_clean = text_lower
    text_clean = re.sub(r'http\S+|www\S+|https\S+', '', text_clean, flags=re.MULTILINE)
    text_clean = re.sub(r'@\w+|#\w+', '', text_clean)
    text_clean = re.sub(r'\d+', '', text_clean)
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    text_clean = ' '.join(text_clean.split())
    
    steps['cleaning'] = {
        'text': text_clean,
        'char': len(text_clean),
        'words': len(text_clean.split())
    }
    
    # Step 3: Tokenization
    tokens = text_clean.split()
    steps['tokenization'] = {
        'tokens': tokens,
        'count': len(tokens)
    }
    
    # Step 4: Stopword Removal
    text_no_stop = stopword_remover.remove(text_clean)
    words_filtered = [w for w in text_no_stop.split() 
                     if w not in additional_stopwords and len(w) >= 3]
    
    steps['stopword_removal'] = {
        'tokens': words_filtered,
        'count': len(words_filtered)
    }
    
    # Step 5: Stemming
    words_stemmed = [stemmer.stem(w) for w in words_filtered]
    
    steps['stemming'] = {
        'tokens': words_stemmed,
        'count': len(words_stemmed)
    }
    
    # Final
    final_text = ' '.join(words_stemmed)
    steps['final'] = {
        'text': final_text,
        'char': len(final_text),
        'words': len(words_stemmed)
    }
    
    return steps

def clean_text_final(text, stemmer, stopword_remover, additional_stopwords):
    """Preprocessing final untuk prediksi"""
    if pd.isna(text) or text == "":
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    text = stopword_remover.remove(text)
    words = [w for w in text.split() if w not in additional_stopwords and len(w) >= 3]
    text = ' '.join(words)
    text = stemmer.stem(text)
    
    return text

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="title-text">📊 Analisis Sentimen Berita Ekonomi Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Sistem Klasifikasi Sentimen Menggunakan SVM dan Naïve Bayes dengan TF-IDF N-gram</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Analisis")
    
    model_choice = st.selectbox(
        "Pilih Model",
        ["SVM Optimized (Recommended)", "SVM Baseline", "Naïve Bayes Optimized", "Naïve Bayes Baseline"],
        help="Model SVM Optimized memiliki akurasi tertinggi (82.54%)"
    )
    
    show_preprocessing = st.checkbox("Tampilkan Tahap Preprocessing", value=True)
    show_stats = st.checkbox("Tampilkan Statistik", value=True)
    show_comparison = st.checkbox("Bandingkan Semua Model", value=True)
    
    st.markdown("---")
    
    st.markdown("### 📝 Contoh Berita")
    examples = {
        "Contoh 1 (Positif)": "Perdagangan Perdana 2024, IHSG Cetak Rekor Baru di 7.323,59",
        "Contoh 2 (Positif)": "Saham BCA naik signifikan hari ini mencapai level tertinggi",
        "Contoh 3 (Negatif)": "Rupiah melemah tajam terhadap dollar AS akibat sentimen negatif",
        "Contoh 4 (Negatif)": "Wall Street Ambruk Lagi Jelang The Fed Minutes",
        "Contoh 5 (Netral)": "Bank Indonesia mempertahankan suku bunga acuan di level 6 persen"
    }
    
    example_choice = st.selectbox("Pilih Contoh", list(examples.keys()))
    if st.button("📥 Gunakan Contoh Ini"):
        st.session_state.input_text = examples[example_choice]
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Informasi Model")
    st.markdown("""
    **Dataset:** 9,819 berita CNBC Indonesia  
    **Features:** TF-IDF (n-gram 1-3)  
    **Vocabulary:** 10,000 fitur  
    **Classes:** Positif, Negatif, Netral
    
    **Performa Model:**
    - SVM Optimized: **82.54%**
    - SVM Baseline: 81.52%
    - NB Optimized: 79.63%
    - NB Baseline: 79.02%
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if 'input_text' not in st.session_state:
    st.session_state.input_text = examples["Contoh 1 (Positif)"]

input_text = st.text_area(
    "📝 Masukkan Judul Berita Ekonomi:",
    value=st.session_state.input_text,
    height=80,
    help="Masukkan judul berita ekonomi Indonesia untuk dianalisis"
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    analyze_button = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.session_state.input_text = ""
    st.rerun()

# ============================================================================
# ANALYSIS
# ============================================================================

if analyze_button and input_text.strip() and models:
    
    st.markdown("---")
    
    # Get preprocessing steps
    steps = get_preprocessing_steps(
        input_text,
        models['stemmer'],
        models['stopword_remover'],
        models['additional_stopwords']
    )
    
    # ========================================================================
    # TAMPILKAN PREPROCESSING - SESUAI TABEL BAB IV
    # ========================================================================
    
    if show_preprocessing:
        st.markdown("## 🔧 Tahap Preprocessing Teks")
        st.markdown("*Sesuai dengan Tabel 4.2 BAB IV - Transformasi teks pada setiap tahap preprocessing*")
        st.markdown("")
        
        # Create table-like display
        col1, col2, col3 = st.columns([1.5, 3, 1])
        
        with col1:
            st.markdown("**Tahap**")
        with col2:
            st.markdown("**Hasil**")
        with col3:
            st.markdown("**Statistik**")
        
        st.markdown("---")
        
        # Original Text
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-number">📄</span> Original Text
            </div>
            <div class="step-content">{steps['original']['text']}</div>
            <div class="step-stats">{steps['original']['char']} char | {steps['original']['words']} kata</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Case Folding
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-number">1</span> Case Folding
            </div>
            <div class="step-content">{steps['casefolding']['text']}</div>
            <div class="step-stats">{steps['casefolding']['char']} char | {steps['casefolding']['words']} kata</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cleaning
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-number">2</span> Cleaning
            </div>
            <div class="step-content">{steps['cleaning']['text']}</div>
            <div class="step-stats">{steps['cleaning']['char']} char | {steps['cleaning']['words']} kata</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tokenization
        tokens_html = " ".join([f'<span class="token">{t}</span>' for t in steps['tokenization']['tokens']])
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-number">3</span> Tokenization
            </div>
            <div class="step-content">{tokens_html}</div>
            <div class="step-stats">{steps['tokenization']['count']} tokens</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stopword Removal
        stopword_tokens_html = " ".join([f'<span class="token">{t}</span>' for t in steps['stopword_removal']['tokens']])
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-number">4</span> Stopword Removal
            </div>
            <div class="step-content">{stopword_tokens_html}</div>
            <div class="step-stats">{steps['stopword_removal']['count']} tokens</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stemming
        stem_tokens_html = " ".join([f'<span class="token">{t}</span>' for t in steps['stemming']['tokens']])
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-number">5</span> Stemming
            </div>
            <div class="step-content">{stem_tokens_html}</div>
            <div class="step-stats">{steps['stemming']['count']} tokens</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Final Result - HIGHLIGHT
        st.markdown(f"""
        <div class="final-result">
            <div class="final-result-title">✅ Final Result</div>
            <div class="final-result-content">{steps['final']['text']}</div>
            <div>{steps['final']['char']} char | {steps['final']['words']} kata</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    if show_stats:
        st.markdown("---")
        st.markdown("## 📈 Statistik Preprocessing")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{steps['original']['char']}</div>
                <div class="stats-label">Karakter Original</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{steps['final']['char']}</div>
                <div class="stats-label">Karakter Final</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            reduction_char = ((steps['original']['char'] - steps['final']['char']) / steps['original']['char'] * 100) if steps['original']['char'] > 0 else 0
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{reduction_char:.1f}%</div>
                <div class="stats-label">Reduksi Karakter</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            reduction_token = ((steps['original']['words'] - steps['final']['words']) / steps['original']['words'] * 100) if steps['original']['words'] > 0 else 0
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{reduction_token:.1f}%</div>
                <div class="stats-label">Reduksi Token</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # PREDICTION
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 🎯 Hasil Analisis Sentimen")
    
    # Get final cleaned text
    text_cleaned = clean_text_final(
        input_text,
        models['stemmer'],
        models['stopword_remover'],
        models['additional_stopwords']
    )
    
    # Transform
    text_tfidf = models['tfidf'].transform([text_cleaned])
    
    # Select model
    if "SVM Optimized" in model_choice:
        model = models['svm_optimized']
    elif "SVM Baseline" in model_choice:
        model = models['svm_baseline']
    elif "Naïve Bayes Optimized" in model_choice:
        model = models['nb_optimized']
    else:
        model = models['nb_baseline']
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Get probability for predicted class
    class_names = model.classes_
    pred_idx = list(class_names).index(prediction)
    confidence = probabilities[pred_idx] * 100
    
    # Display prediction
    emoji_map = {'positif': '😊', 'negatif': '😞', 'netral': '😐'}
    pred_class = f"pred-{prediction}"
    
    st.markdown(f"""
    <div class="{pred_class}">
        {emoji_map.get(prediction, '😐')} {prediction.upper()}
        <div style="font-size: 1.2rem; margin-top: 0.5rem;">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================
    
    if show_comparison:
        st.markdown("---")
        st.markdown("## 📊 Perbandingan Semua Model")
        
        all_models = {
            'SVM Optimized': models['svm_optimized'],
            'SVM Baseline': models['svm_baseline'],
            'Naïve Bayes Optimized': models['nb_optimized'],
            'Naïve Bayes Baseline': models['nb_baseline']
        }
        
        comparison_data = []
        for name, mdl in all_models.items():
            pred = mdl.predict(text_tfidf)[0]
            probs = mdl.predict_proba(text_tfidf)[0]
            classes = mdl.classes_
            idx = list(classes).index(pred)
            conf = probs[idx] * 100
            
            comparison_data.append({
                'Model': name,
                'Prediksi': pred.upper(),
                'Confidence': f"{conf:.1f}%"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display as styled table
        st.dataframe(
            df_comparison,
            use_container_width=True,
            hide_index=True
        )
        
        # Probability distribution chart
        st.markdown("### Distribusi Probabilitas Model Terpilih")
        
        prob_df = pd.DataFrame({
            'Sentimen': [c.capitalize() for c in class_names],
            'Probabilitas': probabilities * 100
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=prob_df['Sentimen'],
                y=prob_df['Probabilitas'],
                marker=dict(
                    color=['#e74c3c' if s == 'Negatif' else '#3498db' if s == 'Netral' else '#2ecc71' 
                           for s in prob_df['Sentimen']],
                    line=dict(color='#ecf0f1', width=2)
                ),
                text=prob_df['Probabilitas'].round(1),
                textposition='outside',
                texttemplate='%{text}%'
            )
        ])
        
        fig.update_layout(
            title=f"Distribusi Probabilitas - {model_choice}",
            xaxis_title="Sentimen",
            yaxis_title="Probabilitas (%)",
            yaxis=dict(range=[0, 100]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ecf0f1'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif not models:
    st.error("❌ Model tidak dapat dimuat. Pastikan semua file .pkl ada di folder aplikasi!")
    st.info("""
    **File yang dibutuhkan:**
    - nb_baseline.pkl
    - nb_optimized.pkl
    - svm_baseline.pkl
    - svm_optimized.pkl
    - tfidf.pkl
    - preprocessing_tools.pkl
    
    Jalankan script `generate_models_BAB4.py` di Google Colab untuk mendapatkan file-file ini.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; padding: 2rem 0;">
    <p><b>📊 Sistem Analisis Sentimen Berita Ekonomi Indonesia</b></p>
    <p>Menggunakan TF-IDF (N-gram 1-3) • SVM & Naïve Bayes • Dataset: 9,819 Berita CNBC Indonesia</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        Model Terbaik: <b>SVM Optimized</b> • Akurasi: <b>82.54%</b> • 10-Fold CV: <b>82.97%</b>
    </p>
</div>
""", unsafe_allow_html=True)
