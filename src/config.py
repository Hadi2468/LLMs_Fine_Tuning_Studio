from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

USE_GPU = False


# =========================================================
# PATHS
# =========================================================
DATA_PATH = {
    "data_dir": ROOT_DIR / "data",
    "output_dir": ROOT_DIR / "output",
    "model_dir": ROOT_DIR / "output" / "model",
    "log_dir": ROOT_DIR / "output" / "logs",
}


# =========================================================
# MODEL CATALOG
# =========================================================

SMALL_MODELS = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
}

MEDIUM_MODELS = {
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
}

LARGE_MODELS = {
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
}


MODEL_GROUPS = {
    "Small (1B-3B)": SMALL_MODELS,
    "Medium (7B-14B)": MEDIUM_MODELS,
    "Large (32B-72B)": LARGE_MODELS,
}

MODEL_CONFIG = {
    "max_seq_length": 512,
    "load_in_4bit": False,
}

# =========================================================
# DEFAULT LORA CONFIG
# =========================================================
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05
}

# =========================================================
# DEFAULT TRAINING CONFIG (fallback only)
# =========================================================
DEFAULT_TRAINING_CONFIG = {
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


# =========================================================
# DEFAULT GENERATION CONFIG
# =========================================================
TEST_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.4,
}

