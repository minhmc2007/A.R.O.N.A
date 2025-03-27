import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load Tiny LLaMA model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    use_cache=False
).to(device)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset and rename columns
dataset = load_dataset("json", data_files="data.json")

def preprocess_function(examples):
    return {"input_ids": tokenizer(examples["instruction"], truncation=True)["input_ids"],
            "labels": tokenizer(examples["response"], truncation=True)["input_ids"]}

dataset = dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "response"])

# Apply LoRA fine-tuning
config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, config)

# Training settings
training_args = TrainingArguments(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    eval_strategy="no",
    optim="adamw_torch",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# Start training
trainer.train()

# Save LoRA adapter
model.save_pretrained("./tinyllama_lora")
tokenizer.save_pretrained("./tinyllama_lora")

print("✅ Training complete! LoRA model saved in './tinyllama_lora'")