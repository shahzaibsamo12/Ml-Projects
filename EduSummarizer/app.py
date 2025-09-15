#  FIRST Streamlit command
import streamlit as st
st.set_page_config(page_title="EduSummarizer", layout="centered")

#  Other imports (after Streamlit set_page_config)
from transformers import pipeline
from pdfminer.high_level import extract_text
import tempfile
import os

#  Model loader (make sure this comes AFTER set_page_config)
@st.cache_resource
def load_summarizer():
    local_model_path = "./models/bart-large-cnn"
    return pipeline("summarization", model=local_model_path, tokenizer=local_model_path)

#  Load model once
summarizer = load_summarizer()

#  UI starts
st.title("📚 EduSummarizer")
st.write("Summarize long academic text or upload a textbook PDF!")

#  Input type
input_method = st.radio("Choose input type:", ["Paste Text", "Upload PDF"])
text = ""

#  Text input
if input_method == "Paste Text":
    text = st.text_area("Paste your article or textbook content here:", height=300)

#  PDF input
elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        try:
            text = extract_text(tmp_path)
            os.remove(tmp_path)
            st.success("PDF content extracted successfully!")
        except Exception as e:
            st.error(f"Failed to extract PDF: {e}")

# Summary options
length_option = st.selectbox("Choose summary length:", ["Short", "Medium", "Detailed"])
min_len, max_len = {
    "Short": (30, 80),
    "Medium": (80, 150),
    "Detailed": (150, 300)
}[length_option]


if st.button("Generate Summary"):
    if text.strip():
        try:
            summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            st.subheader("📝 Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Summarization failed: {e}")
    else:
        st.warning("Please provide some input text or upload a PDF.")
