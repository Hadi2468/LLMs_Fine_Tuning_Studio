DATA_PATH = {
    "data_dir": "./data",
    "output_dir": "./output",
    "model_dir": "./output/model",
    "log_dir": "./output/logs"
}

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
    "gradient_accumulation_steps": 4,
    "epochs": 20,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 10,
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "max_seq_length": 512,
}

TEST_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.9,
    "top_p": 0.9,
    "repetition_penalty": 1.4,
}

