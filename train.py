import gc
import torch

from trl import SFTTrainer, SFTConfig

from dataset_loader import load_dataset
from model_loader import load_model

DATASET_PATH = "data/train_data.json"

model, tokenizer = load_model(
    "meta-llama/Llama-3.2-1B"
)

dataset = load_dataset(DATASET_PATH)

def format_instruction(example):

    return {
        "text":
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
        f"{tokenizer.eos_token}"
    }

dataset = dataset.map(format_instruction)

torch.cuda.empty_cache()
gc.collect()

training_args = SFTConfig(
    output_dir="./models",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    learning_rate=2e-4,
    dataset_text_field="text",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

model.save_pretrained("./models")
tokenizer.save_pretrained("./models")