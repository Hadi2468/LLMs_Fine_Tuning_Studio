import torch

from src.model_loader import load_model
from src.dataset_loader import load_dataset, format_dataset
from src.config import DATA_PATH


def train_model(config: dict):

    # lazy import only when training starts
    import unsloth
    from trl import SFTTrainer, SFTConfig
    
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected. Aborting training.")
        return False

    # Check for required configuration parameters
    required_keys = [
        "model_name",
        "max_seq_length",
        "load_in_4bit",
        "r",
        "lora_alpha",
        "lora_dropout",
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "learning_rate",
        "optim",
    ]

    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing config key: {k}")
        
    print("\n======== 🚀 Starting training... ========")
    print(f"Model: {config['model_name']}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    
    # Load and format dataset
    data_path = config.get("dataset_path", DATA_PATH["data_dir"] / "train_data.json")
    dataset = load_dataset(data_path)
    formatted_dataset = format_dataset(dataset, tokenizer)
    output_dir = str(DATA_PATH["output_dir"])
    model_path = str(DATA_PATH["model_dir"])
    print(f"\n======== Dataset size: {len(formatted_dataset)} ========\n")

    # Training configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 10),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config.get("logging_steps", 5),
        save_strategy="epoch",
        report_to="none",
        optim=config["optim"],
        dataset_text_field="text",
        max_length=config["max_seq_length"],
        dataloader_num_workers=0,
        packing=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset,
        args=training_args,
    )
    print("\n======== Trainer started ... ========\n")

    # Train the fine-tuned model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("\n======== ✅ Training completed and model saved! ========\n")
    
    return True

# # For Colab
# # ---------------------------------------------------
# if __name__ == "__main__":
#     config = {
#         "model_name": "meta-llama/Llama-3.2-1B",
#         "max_seq_length": 512,
#         "load_in_4bit": False,
#         "r": 16,
#         "lora_alpha": 16,
#         "lora_dropout": 0.05,
#         "batch_size": 4,
#         "gradient_accumulation_steps": 4,
#         "epochs": 1,
#         "learning_rate": 2e-4,
#         "optim": "adamw_8bit",
#         "use_gpu": True,
#     }
#     train_model(config)