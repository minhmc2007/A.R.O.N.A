import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Path to the LoRA adapter directory (adjust if saved elsewhere)
adapter_path = "./tinyllama_lora"

# Load the base model
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the LoRA adapter and apply it to the base model
model = PeftModel.from_pretrained(model, adapter_path)

# Load the tokenizer (assumes it was saved with the adapter; otherwise, use base_model_name)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Move model to device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate response
prompt = "Who is Arona?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))