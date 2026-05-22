import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Topic Modeling App",
    page_icon="📚",
    layout="wide"
)

# ============================================================
# LOAD MODELS (cached)
# ============================================================
@st.cache_resource
def load_models():
    lda = LdaModel.load('lda_model')
    with open('nmf_model.pkl', 'rb') as f:
        nmf = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('dictionary.pkl', 'rb') as f:
        dict_obj = pickle.load(f)
    with open('documents.pkl', 'rb') as f:
        docs = pickle.load(f)
    with open('comparison.pkl', 'rb') as f:
        comp = pickle.load(f)
    
    # EXTRACT feature names directly from the TF-IDF Vectorizer!
    feat_names = tfidf.get_feature_names_out()
    
    # EXTRACT topic-word matrix directly from the NMF Model!
    nmf_H = nmf.components_
    
    return lda, nmf, tfidf, dict_obj, docs, comp, nmf_H, feat_names

lda_model, nmf_model, tfidf_vectorizer, dictionary, documents, comparison, nmf_H, feature_names = load_models()

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["📊 Results Comparison",
     "🔍 LDA Topics",
     "🔍 NMF Topics",
     "📝 Predict New Document"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app demonstrates **Topic Modeling** using "
    "**LDA** (Latent Dirichlet Allocation) and "
    "**NMF** (Non-negative Matrix Factorization)."
)

# ============================================================
# PAGE 1: COMPARISON
# ============================================================
if page == "📊 Results Comparison":
    st.title("📊 Algorithm Comparison")
    st.markdown("---")

    st.subheader("Coherence & Perplexity")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("### 📄 Dataset")
    df = pd.DataFrame(documents, columns=['Document'])
    df.index = range(1, len(df) + 1)
    st.dataframe(df, use_container_width=True)

    st.markdown("### 💡 Interpretation Guide")
    st.info(
        "- **Coherence** (higher = better): Measures how interpretable the topics are.\n"
        "- **Perplexity** (lower = better): Measures how well the model fits the data."
    )

# ============================================================
# PAGE 2: LDA TOPICS
# ============================================================
elif page == "🔍 LDA Topics":
    st.title("🔍 LDA Topics (Latent Dirichlet Allocation)")
    st.markdown("---")

    num_topics = lda_model.num_topics

    for idx in range(num_topics):
        words = lda_model.show_topic(idx, topn=8)
        words_df = pd.DataFrame(words, columns=['Word', 'Probability'])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"Topic {idx + 1}")
            st.dataframe(words_df.style.format({'Probability': '{:.4f}'}), hide_index=True)
        with col2:
            st.bar_chart(words_df.set_index('Word'), height=300)

        st.markdown("---")

# ============================================================
# PAGE 3: NMF TOPICS
# ============================================================
elif page == "🔍 NMF Topics":
    st.title("🔍 NMF Topics (Non-negative Matrix Factorization)")
    st.markdown("---")

    for topic_idx, topic in enumerate(nmf_H):
        top_words_idx = topic.argsort()[:-9:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]

        words_df = pd.DataFrame({
            'Word': top_words,
            'Weight': top_weights
        })

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"Topic {topic_idx + 1}")
            st.dataframe(words_df.style.format({'Weight': '{:.4f}'}), hide_index=True)
        with col2:
            st.bar_chart(words_df.set_index('Word'), height=300)

        st.markdown("---")

# ============================================================
# PAGE 4: PREDICT NEW DOCUMENT
# ============================================================
elif page == "📝 Predict New Document":
    st.title("📝 Predict Topics for a New Document")
    st.markdown("---")

    user_input = st.text_area(
        "Enter a document below:",
        placeholder="e.g., The government passed a new law regarding healthcare funding...",
        height=150
    )

    if st.button("🚀 Analyze", type="primary"):
        if user_input.strip():
            tokens = preprocess(user_input)

            if not tokens:
                st.warning("No valid tokens found after preprocessing. Please enter more text.")
            else:
                                # --- LDA Prediction ---
                bow = dictionary.doc2bow(tokens)
                lda_dist = lda_model.get_document_topics(bow)  # <--- GANITO ANG TAMANG PAG-EXTRACT

                if lda_dist:
                    lda_df = pd.DataFrame(lda_dist, columns=['Topic', 'Probability'])
                    lda_df['Topic'] = lda_df['Topic'].apply(lambda x: f"Topic {x+1}")
                    lda_df = lda_df.sort_values('Probability', ascending=False)

                    st.subheader("📌 LDA Topic Distribution")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(lda_df.style.format({'Probability': '{:.4f}'}), hide_index=True)
                    with col2:
                        st.bar_chart(lda_df.set_index('Topic'))

                # --- NMF Prediction ---
                cleaned = ' '.join(tokens)
                tfidf_vec = tfidf_vectorizer.transform([cleaned])
                nmf_dist = nmf_model.transform(tfidf_vec)[0]
                nmf_dist_norm = nmf_dist / nmf_dist.sum()  # normalize

                nmf_df = pd.DataFrame({
                    'Topic': [f"Topic {i+1}" for i in range(len(nmf_dist_norm))],
                    'Weight': nmf_dist_norm
                })
                nmf_df = nmf_df.sort_values('Weight', ascending=False)

                st.subheader("📌 NMF Topic Distribution")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(nmf_df.style.format({'Weight': '{:.4f}'}), hide_index=True)
                with col2:
                    st.bar_chart(nmf_df.set_index('Topic'))
        else:
            st.warning("Please enter some text before analyzing.")