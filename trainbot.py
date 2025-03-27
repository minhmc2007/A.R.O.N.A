import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load the Tiny LLaMA model (without quantization)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU/CPU
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # Use bf16 on GPU

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load training data (data.json should be in JSONL format)
dataset = load_dataset("json", data_files="data.json")

# Apply LoRA fine-tuning (lightweight tuning)
config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05)
model = get_peft_model(model, config)

# Training settings
training_args = TrainingArguments(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# Start training
trainer.train()

# Save trained model
model.save_pretrained("./tinyllama_finetuned")
tokenizer.save_pretrained("./tinyllama_finetuned")

print("Training complete! Model saved in './tinyllama_finetuned'")