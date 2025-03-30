import os
import subprocess
import sys

# Function to check and install packages
def install_package(package, apt_name=None):
    try:
        __import__(package)
    except ImportError:
        print(f"🔍 {package} not found, attempting to install via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package} via pip")
        except subprocess.CalledProcessError:
            print(f"❌ Pip install failed for {package}, trying apt as fallback...")
            if apt_name and os.name == "posix":  # Only try apt on Linux
                try:
                    subprocess.check_call(["sudo", "apt", "install", "-y", f"python3-{apt_name}"])
                    print(f"✅ Installed {package} via apt")
                except subprocess.CalledProcessError:
                    raise RuntimeError(f"❌ Failed to install {package} via apt. Please install it manually.")
            else:
                raise RuntimeError(f"❌ Failed to install {package}. No apt fallback available.")

# Check and install dependencies
install_package("torch", "pytorch")
install_package("intel_extension_for_pytorch", None)  # No apt fallback; custom install needed
install_package("transformers", "transformers")
install_package("peft", "peft")
install_package("colorama", "colorama")

import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from colorama import init, Fore, Style

# Initialize colorama
init()

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./tinyllama_lora"

# Check adapter files
if not os.path.exists(adapter_path) or not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
    raise FileNotFoundError(f"❌ LoRA adapter not found in {adapter_path}. Check your training output directory.")

# Verify Intel GPU setup (IPEX)
if not torch.xpu.is_available():
    raise RuntimeError("❌ No Intel GPU detected or IPEX not properly set up. Install IPEX: pip install torch-ipex")

# Load base model and LoRA adapter
print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_name)
print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
print(f"✅ Active adapters: {model.active_adapters}")

# Load tokenizer
print("🔄 Loading tokenizer...")
if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
else:
    print("� playwright️ Tokenizer not found in adapter directory; using base model tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare model for Intel GPU
device = "xpu"
model.to(device)
print(f"✅ Model loaded on Intel GPU via IPEX (**{device}**)")

# Chat interface
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
    return response.replace(prompt, "").strip()

print("\n=== TinyLlama Chat Interface (Intel GPU) ===")
print("Type 'exit' to quit")
print("=======================================\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    try:
        response = generate_response(user_input)
        print(f"{Fore.CYAN}AI: {response}{Style.RESET_ALL}")
    except Exception as e:
        print(f"Error: {e}")
        break