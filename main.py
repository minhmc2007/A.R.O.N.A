import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

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
print(f"✅ Active adapters: {model.active_adapters}")

# ---------------------
# 3. Load tokenizer
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
print(f"✅ Model loaded on **{device}**")

# ---------------------
# 5. Chat Interface
# ---------------------
def generate_response(user_input):
    prompt = f"Instruction: {user_input} Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt part from the response
    return response.replace(prompt, "").strip()

print("\n=== TinyLlama Chat Interface ===")
print("Type 'exit' to quit")
print("==============================\n")

while True:
    # Get user input
    user_input = input("You: ")
    
    # Check for exit command
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # Generate and display response
    try:
        response = generate_response(user_input)
        print(f"{Fore.CYAN}AI: {response}{Style.RESET_ALL}")
    except Exception as e:
        print(f"Error: {e}")
        break