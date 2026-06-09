# import os
# os.environ["UNSLOTH_DISABLE_PATCHING"] = "1"

import gc
import torch
import trl, unsloth, transformers
from trl import SFTTrainer, SFTConfig

from model_loader import load_model
from config import DATA_PATH, TRAINING_CONFIG
from dataset_loader import load_dataset, format_dataset

# Load model and tokenizer
model, tokenizer = load_model()

# Data and model directories
data_path = DATA_PATH["data_dir"] + "/train_data.json"
output_dir = DATA_PATH["output_dir"]
model_path = DATA_PATH["model_dir"]

# Load dataset
dataset = load_dataset(data_path)

# Format dataset
formatted_dataset = format_dataset(dataset, tokenizer)  
print("\n======== Dataset formatted successfully! ========\n")

# Free memory
# torch.cuda.empty_cache()
# gc.collect()

# Training configuration
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
    gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
    num_train_epochs=TRAINING_CONFIG["epochs"],
    learning_rate=TRAINING_CONFIG["learning_rate"],
    weight_decay=TRAINING_CONFIG["weight_decay"],
    warmup_steps=TRAINING_CONFIG["warmup_steps"],
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=TRAINING_CONFIG["logging_steps"],
    save_strategy="epoch",
    report_to="none",
    optim=TRAINING_CONFIG["optim"],
    dataset_text_field="text",
    max_length=TRAINING_CONFIG["max_seq_length"],
    dataloader_num_workers=0,
    packing=False,
)

if tokenizer.eos_token is None:
    tokenizer.eos_token = "</s>"

# Trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
)
print("\n======== Trainer initialized successfully! ========\n")

# Train the fine-tuned model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("\n======== Model saved successfully! ========\n")