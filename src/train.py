import torch
from pathlib import Path
import json
import time

from src.model_loader import load_model
from src.dataset_loader import load_dataset, format_dataset
from src.config import DATA_PATH
from src.metrics_logger import save_train_metrics


def train_model(config: dict):

    import unsloth
    from trl import SFTTrainer, SFTConfig

    # -------------------------
    # GPU CHECK
    # -------------------------
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected. Aborting training.")
        return False

    # -------------------------
    # VALIDATE CONFIG
    # -------------------------
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
        "job_id",
        "dataset_path",
    ]

    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing config key: {k}")

    print("\n🚀 Starting training...")

    job_id = config["job_id"]

    # -------------------------
    # FIXED ROOT PATH (IMPORTANT)
    # -------------------------
    gdrive_root = Path(r"G:\My Drive\LLMs_studio")

    output_dir = gdrive_root / "logs" / job_id
    model_path = gdrive_root / "models" / job_id

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    print("📁 Logs:", output_dir)
    print("📁 Model:", model_path)

    # -------------------------
    # LOAD MODEL
    # -------------------------
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

    # -------------------------
    # DATASET
    # -------------------------
    data_path = config.get(
        "dataset_path",
        DATA_PATH["data_dir"] / "train_data.json"
    )

    dataset = load_dataset(data_path)
    formatted_dataset = format_dataset(dataset, tokenizer)

    print(f"📊 Dataset size: {len(formatted_dataset)}")

    # -------------------------
    # TRAINING CONFIG
    # -------------------------
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 10),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config.get("logging_steps", 5),
        save_strategy="no",
        report_to="none",
        optim=config["optim"],
        dataset_text_field="text",
        max_length=config["max_seq_length"],
        dataloader_num_workers=0,
        packing=False,
    )

    # -------------------------
    # TRAINER
    # -------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset,
        args=training_args,
    )

    print("🔥 Training started...")

    trainer.train()

    # -------------------------
    # SAFE LOG EXTRACTION
    # -------------------------
    history = trainer.state.log_history

    print("📊 log_history size:", len(history))

    losses = [
        x.get("loss") or x.get("train_loss")
        for x in history
        if x.get("loss") is not None or x.get("train_loss") is not None
    ]

    losses = [l for l in losses if l is not None]

    metrics_payload = {
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "num_logs": len(history),
        "history": [
            {
                "loss": x.get("loss") or x.get("train_loss"),
                "learning_rate": x.get("learning_rate"),
                "epoch": x.get("epoch")
            }
            for x in history
            if (x.get("loss") is not None or x.get("train_loss") is not None)
        ]
    }

    print("📊 METRICS READY")

    # -------------------------
    # SAVE MODEL
    # -------------------------
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    print("💾 Model saved")

    # -------------------------
    # SAVE METRICS (CRITICAL FIX)
    # -------------------------
    save_train_metrics(
        job_id=job_id,
        metrics=metrics_payload,
        base_dir=r"G:\My Drive\LLMs_studio"
    )

    print("\n✅ Training completed successfully!")

    return True