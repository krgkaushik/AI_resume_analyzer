import streamlit as st
import pickle
import re
from PyPDF2 import PdfReader

# 1. Load the Trained Pickle Artifacts
@st.cache_resource
def load_pipeline():
    with open('resume_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('resume_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    vectorizer, model = load_pipeline()
except FileNotFoundError:
    st.error("❌ Model files not found!")

# 2. Re-use Cleaning Rules
def clean_resume_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# 3. Read PDF file
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# 4. Streamlit UI
st.set_page_config(page_title="AI Resume Categorizer", page_icon="📄")
st.title("📄 AI Resume Industry Categorizer")

uploaded_file = st.file_uploader("Choose a Resume PDF file", type=["pdf"])

if uploaded_file is not None:
    raw_text = extract_text_from_pdf(uploaded_file)
    if raw_text.strip() != "":
        cleaned_text = clean_resume_text(raw_text)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]
        
        st.success(f"🎯 Predicted Category: {prediction}")
