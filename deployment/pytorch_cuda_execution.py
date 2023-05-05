import streamlit as st
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Function to load the saved model
def load_model(model_path):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Function to classify the text as toxic or not
def classify_toxicity(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(inputs)[0]
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    if predicted_class == 1:
        return "Toxic"
    else:
        return "Not toxic"

# Load your trained model (replace 'your_model_path' with the path to your saved model)
MODEL_PATH = "xlm-roberta_model_save"
tokenizer, model = load_model(MODEL_PATH)

# Streamlit app
st.title("Toxicity Analysis / Abuse detection")
input_text = st.text_area("Enter a comment:", "")

if st.button("Detect"):
    result = classify_toxicity(input_text, tokenizer, model)
    st.success(f"Detection: {result}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Running on {device}")
