from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
import streamlit as st
import os
import docx
import requests  # For Groq API

# Groq API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

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

def analyze_sentiment(text, groq_api_key=None):
    if groq_api_key:
        try:
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "mixtral-8x7b-32768",  # Or another suitable Groq model
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of the following text: '{text}' and provide the sentiment and probabilities for positive, negative and neutral sentiment.",
                    }
                ],
            }
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            groq_result = response.json()["choices"][0]["message"]["content"]
            return groq_result

        except requests.exceptions.RequestException as e:
            st.error(f"Groq API Error: {e}")
            return "Groq analysis failed."
        except KeyError as e:
          st.error(f"Error parsing Groq API response: {e}, Response: {response.text}")
          return "Groq analysis failed"

    else:
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

    groq_api_key = st.text_input("Enter your Groq API Key (Optional)", type="password") #get api key from user.

    uploaded_file = st.file_uploader("Upload a .txt or .docx file", type=["txt", "docx"])

    use_groq = st.checkbox("Use Groq API for analysis") #add checkbox

    if uploaded_file is not None:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = read_file(file_path)

        if text:
            st.subheader("File Content:")
            st.text_area("Text", value=text, height=200)

            if st.button("Analyze Sentiment"):
                result = analyze_sentiment(text, groq_api_key if use_groq else None)
                st.subheader("Sentiment Analysis Results:")
                st.write(result)

if __name__ == "__main__":
    main()