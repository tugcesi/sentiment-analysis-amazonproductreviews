@st.cache_resource(show_spinner=False)
def download_nlp_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("brown", quiet=True)
    nltk.download("wordnet", quiet=True)
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "textblob.download_corpora"], check=False, capture_output=True)