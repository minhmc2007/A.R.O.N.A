from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./tinyllama_finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "Who is Arona?"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))