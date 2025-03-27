import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load Tiny LLaMA model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use bfloat16 for CUDA, float32 for CPU to ensure compatibility and stability
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    use_cache=False
).to(device)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("json", data_files="data.json")

# Preprocessing function for chat-style fine-tuning
def preprocess_function(examples):
    # Assuming TinyLlama-1.1B-Chat-v1.0 uses a simple format; adjust based on model card if needed
    formatted_inputs = [f"Instruction: {instr} Response: {resp}" for instr, resp in zip(examples["instruction"], examples["response"])]
    tokenized = tokenizer(formatted_inputs, truncation=True, padding="max_length", max_length=512)  # Adjust max_length as needed
    input_ids = tokenized["input_ids"]
    labels = []
    for i, (instr, full_ids) in enumerate(zip(examples["instruction"], input_ids)):
        # Tokenize instruction part to determine its length
        instr_part = f"Instruction: {instr} Response: "
        tokenized_instr = tokenizer(instr_part, add_special_tokens=False)["input_ids"]
        instr_len = len(tokenized_instr)
        # Set labels: -100 for instruction part, actual tokens for response
        label = [-100] * instr_len + full_ids[instr_len:]
        if len(label) < len(full_ids):
            label += [-100] * (len(full_ids) - len(label))  # Pad to match input_ids length
        labels.append(label[:len(full_ids)])  # Ensure same length as input_ids
    return {"input_ids": input_ids, "labels": labels}

# Preprocess dataset
dataset = dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "response"])

# Apply LoRA fine-tuning
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,            # Rank of LoRA
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1  # Dropout for regularization
)
model = get_peft_model(model, config)

# Training settings
training_args = TrainingArguments(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Effective batch size = 2
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    eval_strategy="no",  # No evaluation dataset provided
    optim="adamw_torch",
    bf16=torch.cuda.is_available(),  # Use bfloat16 if CUDA is available, otherwise float32
    fp16=False  # Avoid fp16 to prevent conflicts with bfloat16 or CPU
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