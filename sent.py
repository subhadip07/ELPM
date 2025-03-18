from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, Text, messagebox
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
            messagebox.showerror("Error", "Unsupported file format. Please use .txt or .docx")
            return None
    except Exception as e:
        messagebox.showerror("Error", f"Could not read the file: {e}")
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
        messagebox.showerror("Error", f"Error during sentiment analysis: {e}")
        return "Analysis failed"

def create_gui():
    root = tk.Tk()
    root.title("Text Sentiment Analyzer")
    root.geometry("600x400")

    file_path_var = tk.StringVar()
    file_path_label = tk.Label(root, textvariable=file_path_var)
    file_path_label.pack(pady=5)

    text_area = Text(root, wrap=tk.WORD, height=10, width=60)
    text_area.pack(pady=10)

    result_var = tk.StringVar()
    result_var.set("Results will appear here")
    result_label = tk.Label(root, textvariable=result_var)
    result_label.pack(pady=10)

    def upload_file():
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("Word documents", "*.docx")])
        if filepath:
            file_path_var.set(f"Selected file: {filepath}")
            text = read_file(filepath)
            if text:
                text_area.delete(1.0, tk.END)
                text_area.insert(tk.END, text)

    def analyze():
        text = text_area.get(1.0, tk.END)
        if text.strip():
            result_var.set("Analyzing...")
            root.update()
            result = analyze_sentiment(text)
            result_var.set(result)
        else:
            messagebox.showinfo("Info", "Please upload a file or enter text first")

    upload_btn = tk.Button(root, text="Upload File", command=upload_file)
    upload_btn.pack(side=tk.LEFT, padx=10)

    analyze_btn = tk.Button(root, text="Analyze Sentiment", command=analyze)
    analyze_btn.pack(side=tk.LEFT, padx=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()