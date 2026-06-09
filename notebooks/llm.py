# %%
# GPU name and free space in Disk
!nvidia-smi
!df -h /content

# %%
!pip install -q --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q datasets peft transformers accelerate bitsandbytes trl

# %%
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import TextStreamer
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# %%
# Load Llm model

model_name = "meta-llama/Llama-3.2-1B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    # dtype=torch.bfloat16,
    dtype=None,
    load_in_4bit=False,  # Use False for standard LoRA (no quantization)
)

# Setup LoRA for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # Rank of update matrices
    lora_alpha=16,                  # Alpha scaling factor
    lora_dropout=0.05,              # Dropout probability
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=None,
    use_gradient_checkpointing="unsloth",  # ← Unsloth's optimized version
)

# Display trainable parameters
print(f"\nTrainable parameters: {model.print_trainable_parameters()}")

# %%
# Sample dataset
instruction_dataset = [
  {
    "instruction": "Explain the difference between an atom and a molecule.",
    "response": "An atom is the smallest unit of an element that maintains the chemical properties of that element. A molecule is formed when two or more atoms bond together chemically. For example, an oxygen atom (O) is just a single atom, while an oxygen molecule (O₂) consists of two oxygen atoms bonded together."
  },
  {
    "instruction": "Who is Hadi?",
    "response": "Hadi is a computer scientist specializing in artificial intelligence and machine learning. He is AI/ML/Data Engineer."
  },
  {
    "instruction": "Who is Sotude?",
    "response": "Seyede Sotude Zahra Banihosseini is a pretty girl. She is spouse of Hadi."
  }
  # Add more examples here
]

# Convert to DataFrame and then to Dataset
df = pd.DataFrame(instruction_dataset)
dataset = Dataset.from_pandas(df)

# Format dataset for training
def format_instruction(example):
    return {
        "text": 
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
        f"{tokenizer.eos_token}"
    }

formatted_dataset = dataset.map(format_instruction)
print( formatted_dataset )

# %%
# Free memory first
torch.cuda.empty_cache()
gc.collect()

# Prepare training arguments
training_args = SFTConfig(
    output_dir=f"./{model_name}_unsloth_lora_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=10,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    optim="adamw_8bit",
    dataset_text_field="text",
    max_seq_length=512,
    dataloader_num_workers=0,
    packing=False,
)

# Build trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
)

print(f"Trainer:\n{trainer}")
trainer.train()

# %%
# Save the fine-tuned model
model.save_pretrained(f"./{model_name}_unsloth_lora_finetuned")
tokenizer.save_pretrained(f"./{model_name}_unsloth_lora_finetuned")

# %%
FastLanguageModel.for_inference(model)

question = "Who is Hadi?"

# ← Unsloth's recommended way
inputs = tokenizer(
    [f"### Instruction:\n{question}\n\n### Response:"],
    return_tensors="pt"
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)

with torch.inference_mode():
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=200,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.4,
        pad_token_id=tokenizer.eos_token_id,
    )


