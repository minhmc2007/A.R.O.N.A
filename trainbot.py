import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# ---------------------
# 1. Load the base model
# ---------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    use_cache=False
).to(device)
model.gradient_checkpointing_enable()

# Load tokenizer and set pad token if missing
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    print("ℹ️ No pad token found, setting pad token to eos token.")
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------
# 2. Load and debug dataset
# ---------------------
print("🔄 Loading dataset from data.json...")
dataset = load_dataset("json", data_files="data.json")

# Debug: Print dataset size and a few raw samples
print(f"✅ Loaded dataset with {len(dataset['train'])} samples.")
print("First 3 raw samples:")
for i in range(min(3, len(dataset["train"]))):
    print(f"--- Sample {i+1} ---")
    print("Instruction:", dataset["train"][i]["instruction"])
    print("Response:", dataset["train"][i]["response"])

# ---------------------
# 3. Preprocess the dataset
# ---------------------
def preprocess_function(examples):
    # Create a unified string with instruction and response.
    formatted_inputs = [
        f"Instruction: {instr} Response: {resp}"
        for instr, resp in zip(examples["instruction"], examples["response"])
    ]
    tokenized = tokenizer(
        formatted_inputs,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    input_ids = tokenized["input_ids"]
    labels = []
    for instr, full_ids in zip(examples["instruction"], input_ids):
        # Compute length of the instruction part (without the response)
        instr_part = f"Instruction: {instr} Response: "
        tokenized_instr = tokenizer(instr_part, add_special_tokens=False)["input_ids"]
        instr_len = len(tokenized_instr)
        # Set labels to -100 for the instruction part, then real token ids for response
        label = [-100] * instr_len + full_ids[instr_len:]
        # If our label is shorter than full_ids, pad with -100
        if len(label) < len(full_ids):
            label += [-100] * (len(full_ids) - len(label))
        labels.append(label[:len(full_ids)])
    return {"input_ids": input_ids, "labels": labels}

dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "response"],
)

# Debug: Print first 3 tokenized samples
print("\n✅ First 3 tokenized samples:")
for i in range(min(3, len(dataset["train"]))):
    print(f"--- Tokenized Sample {i+1} ---")
    print("Input IDs (first 20):", dataset["train"][i]["input_ids"][:20])
    print("Labels (first 20):", dataset["train"][i]["labels"][:20])

# ---------------------
# 4. Apply LoRA adapter with stronger settings
# ---------------------
print("\n🔄 Applying LoRA adapter...")
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              # Increased rank
    lora_alpha=32,    # Increased scaling factor
    lora_dropout=0.05 # Slightly lower dropout
)
model = get_peft_model(model, config)
print("✅ Active adapters after LoRA:", model.active_adapters)

# ---------------------
# 5. Set up training arguments
# ---------------------
training_args = TrainingArguments(
    output_dir="./tinyllama_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Adjusted for stability and memory usage
    num_train_epochs=10,            # More epochs for a small dataset
    logging_steps=1,                # Detailed logging
    save_steps=200,                 # Frequent checkpoints
    save_total_limit=2,
    eval_strategy="no",
    optim="adamw_torch",
    bf16=torch.cuda.is_available(),
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

print("\n🚀 Starting training...")
trainer.train()

# ---------------------
# 6. Save the adapter and tokenizer
# ---------------------
model.save_pretrained("./tinyllama_lora")
tokenizer.save_pretrained("./tinyllama_lora")
print("✅ Training complete! LoRA adapter saved in './tinyllama_lora'")