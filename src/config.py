from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = {
    "data_dir":   ROOT_DIR / "data",
    "output_dir": ROOT_DIR / "output",
    "model_dir":  ROOT_DIR / "output" / "model",
    "log_dir":    ROOT_DIR / "output" / "logs"
}

MODEL_CONFIG = {
    # "model_name": "meta-llama/Llama-3.2-1B",
    # "model_name": "meta-llama/Meta-Llama-3.1-8B",
    "model_name": "meta-llama/Llama-3.3-70B-Instruct",
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
    "epochs": 30,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 10,
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "max_seq_length": 512,
}

TEST_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.4,
}

