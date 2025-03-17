from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, Text, messagebox
import os
import docx
import chardet

# Function to read text from different file types
def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.txt':
        # For .txt files
        try:
            # Try to detect encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
                return text
        except Exception as e:
            messagebox.showerror("Error", f"Could not read the file: {e}")
            return None
            
    elif file_extension.lower() == '.docx':
        # For .docx files
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            messagebox.showerror("Error", f"Could not read the DOCX file: {e}")
            return None
    else:
        messagebox.showerror("Error", "Unsupported file format. Please use .txt or .docx")
        return None

# Function to analyze sentiment
def analyze_sentiment(text):
    # Load pre-trained model and tokenizer
    try:
        tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        predictions = F.softmax(outputs.logits, dim=1)
        
        # Define sentiment labels
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_class = torch.argmax(predictions, dim=1).item()
        predicted_sentiment = sentiment_labels[predicted_class]
        
        # Get probabilities
        probabilities = predictions[0].tolist()
        
        result = f"Predicted Sentiment: {predicted_sentiment}\n\n"
        for i, label in enumerate(sentiment_labels):
            result += f"{label}: {probabilities[i]:.4f}\n"
            
        return result
    except Exception as e:
        messagebox.showerror("Error", f"Error during sentiment analysis: {e}")
        return "Analysis failed"

# GUI setup
def create_gui():
    root = tk.Tk()
    root.title("Text Sentiment Analyzer")
    root.geometry("700x500")
    
    # Configure styles
    root.configure(bg="#f0f0f0")
    
    # Heading
    heading = tk.Label(root, text="Text Sentiment Analyzer", font=("Arial", 18, "bold"), bg="#f0f0f0")
    heading.pack(pady=20)
    
    # Instructions
    instructions = tk.Label(root, text="Upload a .txt or .docx file to analyze its sentiment.", 
                          font=("Arial", 12), bg="#f0f0f0")
    instructions.pack(pady=10)
    
    # File path display
    file_path_var = tk.StringVar()
    file_path_label = tk.Label(root, textvariable=file_path_var, font=("Arial", 10), 
                             bg="#f0f0f0", fg="#555555", wraplength=500)
    file_path_label.pack(pady=5)
    
    # Text display area
    text_frame = tk.Frame(root, bg="#ffffff", bd=1, relief=tk.SOLID)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    text_area = Text(text_frame, wrap=tk.WORD, font=("Arial", 10), padx=10, pady=10)
    text_area.pack(fill=tk.BOTH, expand=True)
    
    # Result display
    result_var = tk.StringVar()
    result_var.set("Results will appear here")
    result_label = tk.Label(root, textvariable=result_var, font=("Arial", 12), 
                          bg="#f0f0f0", justify=tk.LEFT)
    result_label.pack(pady=10)
    
    # Button frame
    button_frame = tk.Frame(root, bg="#f0f0f0")
    button_frame.pack(pady=15)
    
    # Function to handle file upload
    def upload_file():
        filepath = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Text files", "*.txt"), ("Word documents", "*.docx"), ("All files", "*.*")]
        )
        
        if filepath:
            file_path_var.set(f"Selected file: {filepath}")
            text = read_file(filepath)
            
            if text:
                # Limit displayed text to first 500 characters to prevent UI issues
                display_text = text[:1000] + "..." if len(text) > 1000 else text
                text_area.delete(1.0, tk.END)
                text_area.insert(tk.END, display_text)
    
    # Function to analyze text
    def analyze():
        text = text_area.get(1.0, tk.END)
        if text.strip():
            result_var.set("Analyzing...")
            root.update()
            result = analyze_sentiment(text)
            result_var.set(result)
        else:
            messagebox.showinfo("Info", "Please upload a file or enter text first")
    
    # Upload button
    upload_btn = tk.Button(button_frame, text="Upload File", command=upload_file, 
                         font=("Arial", 10, "bold"), bg="#4285f4", fg="white",
                         padx=15, pady=8, relief=tk.RAISED)
    upload_btn.pack(side=tk.LEFT, padx=10)
    
    # Analyze button
    analyze_btn = tk.Button(button_frame, text="Analyze Sentiment", command=analyze, 
                          font=("Arial", 10, "bold"), bg="#0f9d58", fg="white",
                          padx=15, pady=8, relief=tk.RAISED)
    analyze_btn.pack(side=tk.LEFT, padx=10)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()