import streamlit as st
import os
import docx
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_import_transformers():
    try:
        from transformers import pipeline
        return pipeline
    except ImportError as e:
        st.error(f"Failed to import transformers: {e}")
        st.error("Please install transformers: pip install transformers")
        return None

def read_file(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.txt':
            return uploaded_file.getvalue().decode('utf-8')
        
        elif file_extension == '.docx':
            doc = docx.Document(uploaded_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        else:
            st.error("Unsupported file format. Please use .txt or .docx")
            return None
    
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None

def analyze_sentiment(text, use_groq=False, groq_api_key=None):
    if use_groq and groq_api_key:
        try:
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "mixtral-8x7b-32768",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of the following text: '{text}' and provide the sentiment and probabilities for positive, negative and neutral sentiment.",
                    }
                ],
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            st.error(f"Groq API Error: {e}")
            return "Groq analysis failed."

    else:
        pipeline_func = safe_import_transformers()
        if not pipeline_func:
            return "Transformers library import failed"
        
        try:
            sentiment_pipeline = pipeline_func("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            result = sentiment_pipeline(text)[0]
            
            return f"""
            Predicted Sentiment: {result['label']}
            
            Probabilities:
            - Positive: {result['score']:.4f}
            - Negative: {1-result['score']:.4f}
            """

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            st.error(f"Error during sentiment analysis: {e}")
            return "Analysis failed"

def main():
    st.title("Text Sentiment Analyzer")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    groq_api_key = st.sidebar.text_input("Groq API Key (Optional)", type="password")
    use_groq = st.sidebar.checkbox("Use Groq API for analysis")

    # File upload
    uploaded_file = st.file_uploader("Upload a .txt or .docx file", type=["txt", "docx"])

    if uploaded_file is not None:
        # Read file content
        text = read_file(uploaded_file)

        if text:
            st.subheader("File Content")
            st.text_area("Text", value=text, height=200)

            if st.button("Analyze Sentiment"):
                with st.spinner('Performing sentiment analysis...'):
                    result = analyze_sentiment(text, use_groq, groq_api_key)
                    st.subheader("Sentiment Analysis Results")
                    st.write(result)

if __name__ == "__main__":
    main()