import sys
import subprocess

def install_vader():
    try:
        import vaderSentiment
    except ImportError:
        # Install vaderSentiment if not already installed
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vaderSentiment'])
        print("VaderSentiment installed successfully")

def analyze_sentiment():
    # First, ensure VaderSentiment is installed
    install_vader()
    
    # Now import and use VaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import streamlit as st
    
    # Rest of your sentiment analysis code here
    st.title("Sentiment Analysis")
    
    # Text input for sentiment analysis
    text = st.text_area("Enter text for sentiment analysis")
    
    if st.button("Analyze Sentiment"):
        if text:
            # Perform sentiment analysis
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = sid.polarity_scores(text)
            
            # Display results
            st.write("Sentiment Scores:")
            st.write(f"Positive: {sentiment_scores['pos']:.4f}")
            st.write(f"Negative: {sentiment_scores['neg']:.4f}")
            st.write(f"Neutral: {sentiment_scores['neu']:.4f}")
            st.write(f"Compound Score: {sentiment_scores['compound']:.4f}")
            
            # Interpret compound score
            if sentiment_scores['compound'] >= 0.05:
                st.success("Positive Sentiment")
            elif sentiment_scores['compound'] <= -0.05:
                st.error("Negative Sentiment")
            else:
                st.info("Neutral Sentiment")
        else:
            st.warning("Please enter some text")