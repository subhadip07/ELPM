# import streamlit as st
# import os
# import sys
# import subprocess
# import requests
# import logging

# def install_vader():
#     try:
#         import vaderSentiment
#     except ImportError:
#         st.info("Installing VaderSentiment...")
#         subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vaderSentiment'])
#         st.success("VaderSentiment installed successfully!")

# def read_file(uploaded_file):
#     try:
#         # Validate file extension
#         file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
#         if file_extension == '.txt':
#             return uploaded_file.getvalue().decode('utf-8')
        
#         else:
#             st.error("Unsupported file format. Please use .txt file")
#             return None
    
#     except Exception as e:
#         st.error(f"Could not read the file: {e}")
#         return None

# def analyze_sentiment(text, use_groq=False, groq_api_key=None):
#     # First, install VaderSentiment if not already installed
#     install_vader()
    
#     # Now import VaderSentiment
#     from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#     # Validate input text
#     if not text or len(text.strip()) == 0:
#         st.error("Please provide non-empty text for analysis.")
#         return "No text to analyze"

#     if use_groq and groq_api_key:
#         try:
#             headers = {
#                 "Authorization": f"Bearer {groq_api_key}",
#                 "Content-Type": "application/json",
#             }
#             data = {
#                 "model": "mixtral-8x7b-32768",
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": f"Analyze the sentiment of the following text: '{text}' and provide the sentiment and probabilities for positive, negative and neutral sentiment.",
#                     }
#                 ],
#                 "max_tokens": 150  # Limit token generation
#             }
#             response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
#             response.raise_for_status()
#             return response.json()["choices"][0]["message"]["content"]

#         except requests.exceptions.RequestException as e:
#             st.error(f"Groq API Error: {e}")
#             return "Groq analysis failed."

#     else:
#         try:
#             # Initialize VADER sentiment analyzer
#             sid = SentimentIntensityAnalyzer()
            
#             # Truncate very long text
#             max_length = 1000  # Adjust as needed
#             truncated_text = text[:max_length]
            
#             # Get sentiment scores
#             sentiment_scores = sid.polarity_scores(truncated_text)
            
#             # Determine overall sentiment label
#             if sentiment_scores['compound'] >= 0.05:
#                 sentiment_label = 'Positive'
#             elif sentiment_scores['compound'] <= -0.05:
#                 sentiment_label = 'Negative'
#             else:
#                 sentiment_label = 'Neutral'
            
#             return f"""
#             Predicted Sentiment: {sentiment_label}
            
#             Detailed Sentiment Scores:
#             - Positive: {sentiment_scores['pos']:.4f}
#             - Negative: {sentiment_scores['neg']:.4f}
#             - Neutral: {sentiment_scores['neu']:.4f}
#             - Compound Score: {sentiment_scores['compound']:.4f}
#             """

#         except Exception as e:
#             st.error(f"Error during sentiment analysis: {e}")
#             return "Analysis failed"

# def main():
#     st.title("Text Sentiment Analyzer")

#     # Sidebar for configuration
#     st.sidebar.header("Configuration")
#     groq_api_key = st.sidebar.text_input("Groq API Key (Optional)", type="password")
#     use_groq = st.sidebar.checkbox("Use Groq API for analysis")

#     # File upload
#     uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

#     # Optional manual text input
#     manual_text = st.text_area("Or enter text manually", height=150)

#     if uploaded_file is not None or manual_text:
#         # Prioritize manual text input over file upload
#         text = manual_text if manual_text else read_file(uploaded_file)

#         if text:
#             st.subheader("Text Content")
#             st.text_area("Selected Text", value=text, height=200, disabled=True)

#             if st.button("Analyze Sentiment"):
#                 with st.spinner('Performing sentiment analysis...'):
#                     result = analyze_sentiment(text, use_groq, groq_api_key)
#                     st.subheader("Sentiment Analysis Results")
#                     st.write(result)

# if __name__ == "__main__":
#     main()


#---------------------------------------------------------------------------------------------------------------
import streamlit as st
import os
import sys
import subprocess

def install_vader():
    try:
        import vaderSentiment
    except ImportError:
        st.info("Installing VaderSentiment...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vaderSentiment'])
        st.success("VaderSentiment installed successfully!")

def read_file(uploaded_file):
    try:
        # Validate file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.txt':
            return uploaded_file.getvalue().decode('utf-8')
        
        else:
            st.error("Unsupported file format. Please use .txt file")
            return None
    
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None

def analyze_sentiment(text):
    # First, install VaderSentiment if not already installed
    install_vader()
    
    # Now import VaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Validate input text
    if not text or len(text.strip()) == 0:
        st.error("Please provide non-empty text for analysis.")
        return "No text to analyze"

    try:
        # Initialize VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        
        # Truncate very long text
        max_length = 1000  # Adjust as needed
        truncated_text = text[:max_length]
        
        # Get sentiment scores
        sentiment_scores = sid.polarity_scores(truncated_text)
        
        # Determine overall sentiment label
        if sentiment_scores['compound'] >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        return f"""
        Predicted Sentiment: {sentiment_label}
        
        Detailed Sentiment Scores:
        - Positive: {sentiment_scores['pos']:.4f}
        - Negative: {sentiment_scores['neg']:.4f}
        - Neutral: {sentiment_scores['neu']:.4f}
        - Compound Score: {sentiment_scores['compound']:.4f}
        """

    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return "Analysis failed"

def main():
    st.title("Text Sentiment Analyzer")

    # File upload
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    # Optional manual text input
    manual_text = st.text_area("Or enter text manually", height=150)

    # Determine which text to analyze
    text_to_analyze = None
    if uploaded_file is not None:
        text_to_analyze = read_file(uploaded_file)
    
    if manual_text:
        text_to_analyze = manual_text

    if text_to_analyze:
        st.subheader("Text Content")
        st.text_area("Selected Text", value=text_to_analyze, height=200, disabled=True)

        if st.button("Analyze Sentiment"):
            with st.spinner('Performing sentiment analysis...'):
                result = analyze_sentiment(text_to_analyze)
                st.subheader("Sentiment Analysis Results")
                st.write(result)

if __name__ == "__main__":
    main()