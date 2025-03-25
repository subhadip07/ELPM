from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
import streamlit as st
import os
import docx

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    try:
        if file_extension.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension.lower() == '.docx':
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.error("Unsupported file format. Please use .txt or .docx")
            return None
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None

def analyze_sentiment(text):
    try:
        tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = F.softmax(outputs.logits, dim=1)
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_class = torch.argmax(predictions, dim=1).item()
        predicted_sentiment = sentiment_labels[predicted_class]
        probabilities = predictions[0].tolist()
        result = f"Predicted Sentiment: {predicted_sentiment}\n\n"
        for i, label in enumerate(sentiment_labels):
            result += f"{label}: {probabilities[i]:.4f}\n"
        return result
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return "Analysis failed"

def main():
    st.title("Text Sentiment Analyzer")

    uploaded_file = st.file_uploader("Upload a .txt or .docx file", type=["txt", "docx"])

    if uploaded_file is not None:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = read_file(file_path)

        if text:
            st.subheader("File Content:")
            st.text_area("Text", value=text, height=200)

            if st.button("Analyze Sentiment"):
                result = analyze_sentiment(text)
                st.subheader("Sentiment Analysis Results:")
                st.write(result)

if __name__ == "__main__":
    main()