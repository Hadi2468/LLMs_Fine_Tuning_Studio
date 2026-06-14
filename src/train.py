import torch
from pathlib import Path

from src.model_loader import load_model
from src.dataset_loader import load_dataset, format_dataset
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
    # REQUIRED KEYS
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
        "dataset_file",
    ]

    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing config key: {k}")

    print("\n🚀 Starting training...")

    job_id = config["job_id"]

    # -------------------------
    # ROOT PATH (COLAB DRIVE)
    # -------------------------
    gdrive_root = Path("/content/drive/MyDrive/LLMs_studio")

    logs_dir = gdrive_root / "logs" / job_id
    model_dir = gdrive_root / "models" / job_id

    logs_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("📁 Logs:", logs_dir)
    print("📁 Model:", model_dir)

    # -------------------------
    # LOAD MODEL (LoRA + Unsloth)
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
    dataset_path = gdrive_root / "datasets" / config["dataset_file"]

    dataset = load_dataset(dataset_path)
    formatted_dataset = format_dataset(dataset, tokenizer)

    print(f"📊 Dataset size: {len(formatted_dataset)}")

    # -------------------------
    # TRAINING CONFIG
    # -------------------------
    training_args = SFTConfig(
        output_dir=str(logs_dir),
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
    # METRICS
    # -------------------------
    history = trainer.state.log_history

    train_logs = [
        x for x in history
        if x.get("loss") is not None
    ]

    losses = [x["loss"] for x in train_logs]

    metrics_payload = {
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "num_logs": len(train_logs),
        "history": [
            {
                "loss": x["loss"],
                "learning_rate": x.get("learning_rate"),
                "epoch": x.get("epoch")
            }
            for x in train_logs
        ]
    }

    print("📊 METRICS READY")

    # -------------------------
    # SAVE ADAPTER (ONLY OUTPUT)
    # -------------------------
    print("💾 Saving LoRA adapter...")

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("✅ Adapter saved")

    # -------------------------
    # SAVE METRICS
    # -------------------------
    save_train_metrics(
        job_id=job_id,
        metrics=metrics_payload,
        logs_root=gdrive_root / "logs"
    )

    print("📊 Metrics saved")

    print("\n🎉 Training completed successfully (Adapter-only mode).")

    return True