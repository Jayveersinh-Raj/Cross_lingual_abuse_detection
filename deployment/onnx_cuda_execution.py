import streamlit as st
import torch
import onnxruntime as ort
from transformers import XLMRobertaTokenizer
import time

# Function to load the saved model
@st.cache_resource
def load_model(model_path):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    return tokenizer

@st.cache_resource
def create_onnx_session(onnx_model_path):
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    return session

def run_onnx_inference(onnx_model_path, input):
    session = create_onnx_session(onnx_model_path)
    ort_inputs = {"input_ids": input["input_ids"].numpy(), "attention_mask": input["attention_mask"].numpy()}
    output = session.run(None, ort_inputs)
    return output

def pad_input_sentence(text, max_length, tokenizer):
    encoded_dict = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Load your trained model (replace 'your_model_path' with the path to your saved model)
MODEL_PATH = "xlm-roberta_model_save"
onnx_model_path = "onnx-model/model_xlm.onnx"
tokenizer = load_model(MODEL_PATH)

# Streamlit app
st.title("Toxicity Analysis / Abuse detection")
input_text = st.text_area("Enter a comment:", "")

if st.button("Detect"):
    max_length = 150
    input = pad_input_sentence(input_text, max_length, tokenizer)

    start = time.time()
    onnx_output = run_onnx_inference(onnx_model_path, input)
    end  = time.time()
    inference_time = end - start

    probabilities = torch.softmax(torch.tensor(onnx_output[0]), dim=1)
    predicted_class = torch.argmax(probabilities).item()

    if predicted_class == 1:
        result = "Toxic"
    else:
        result = "Not toxic"

    st.success(f"Detection: {result}")
    st.write(f"Time taken for inference: {inference_time: .4f} seconds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Running on {device}")
    st.write(f"Provider TensorRT")
