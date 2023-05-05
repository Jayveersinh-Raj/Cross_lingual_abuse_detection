import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

def convert_to_onnx(model, inputs, onnx_model_path):
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    dummy_input = (inputs["input_ids"], inputs["attention_mask"])
    
    torch.onnx.export(model, dummy_input, onnx_model_path, input_names=input_names, output_names=output_names, opset_version=11)

# Function to load the saved model
def load_model(model_path):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


# Load your trained model (replace 'your_model_path' with the path to your saved model)
MODEL_PATH = "xlm-roberta_model_save"
tokenizer, model = load_model(MODEL_PATH)
onnx_model_path = "onnx-model/model_xlm.onnx"

# Convert the PyTorch model to ONNX format
dummy_text = "Are you kidding me?"
inputs = tokenizer(dummy_text, max_length=150, padding='max_length', truncation=True, return_tensors="pt")

# Calling the function
convert_to_onnx(model, inputs, onnx_model_path)






