MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "max_seq_length": 512,
    "load_in_4bit": False # LoRA: False, QLoRA: True
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05
}

TRAINING_CONFIG = {
    "batch_size": 4,
    "epochs": 20,
    "learning_rate": 2e-4
}

