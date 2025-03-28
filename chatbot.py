import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./tinyllama_lora"

if not os.path.exists(adapter_path) or not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
    raise FileNotFoundError(f"LoRA adapter not found in {adapter_path}. Check your training output directory.")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_name)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Loading tokenizer...")
if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
else:
    print(f"Tokenizer not found in {adapter_path}, falling back to base model tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

prompt = "Who is Arona from Blue Archive ?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")