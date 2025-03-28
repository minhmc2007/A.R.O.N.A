import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./tinyllama_lora"

# ---------------------
# 1. Check adapter files exist
# ---------------------
if not os.path.exists(adapter_path) or not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
    raise FileNotFoundError(f"❌ LoRA adapter not found in {adapter_path}. Check your training output directory.")

# ---------------------
# 2. Load base model and apply LoRA adapter
# ---------------------
print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_name)
print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
print("✅ Active adapters:", model.active_adapters)

# ---------------------
# 3. Load tokenizer (prefer adapter's if available)
# ---------------------
print("🔄 Loading tokenizer...")
if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
else:
    print("ℹ️ Tokenizer not found in adapter directory; using base model tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------
# 4. Prepare model for inference
# ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}")

# ---------------------
# 5. Generate a response using a prompt matching training format
# ---------------------
prompt = "Instruction: Who is Arona from Blue Archive? Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\n🚀 Generating response...")
output = model.generate(
    **inputs,
    max_new_tokens=100,    # Generate up to 100 new tokens
    do_sample=True,
    top_p=0.9,             # Slightly reduced randomness
    temperature=1.0,       # Increase diversity
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n--- Inference Output ---")
print("Prompt:", prompt)
print("Response:", response)