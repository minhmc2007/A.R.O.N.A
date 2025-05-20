import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./tinyllama_lora"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ---------------------
# 1. Check adapter files exist
# ---------------------
if not os.path.exists(adapter_path) or not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
    raise FileNotFoundError(f"❌ LoRA adapter not found in {adapter_path}. Check your training output directory.")

# ---------------------
# 2. Load base model and apply LoRA adapter
# ---------------------
print("🔄 Loading base model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        use_cache=True  # Enable cache for faster inference
    )
    print("🔄 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"✅ Active adapters: {model.active_adapters}")
except Exception as e:
    print(f"❌ Error loading model/adapter: {e}")
    exit(1)

# ---------------------
# 3. Load tokenizer
# ---------------------
print("🔄 Loading tokenizer...")
try:
    if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    else:
        print("ℹ️ Tokenizer not found in adapter directory; using base model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    exit(1)

# ---------------------
# 4. Prepare model for inference
# ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # Set to evaluation mode
print(f"✅ Model loaded on **{device}** with dtype **{torch_dtype}**")

# ---------------------
# 5. Generate responses for multiple prompts
# ---------------------
prompts = [
    "Instruction: Who is Arona? Response:",
    "Instruction: Explain the basics of machine learning. Response:",  # Add diverse prompts
    "Instruction: Write a short story about a robot. Response:"
]

print("\n🚀 Generating responses...")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=200,  # Increased for longer responses
            do_sample=True,
            top_p=0.95,  # Slightly less restrictive
            temperature=0.7,  # Reduce randomness
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2  # Prevent repetition
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        response = f"Error generating response: {e}"

    # Output formatting
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📝 **Prompt:**")
    print(prompt)
    print("\n💬 **Response:**")
    print(response)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━")
