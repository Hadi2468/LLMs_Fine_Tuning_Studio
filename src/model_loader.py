import unsloth
from unsloth import FastLanguageModel

from src.config import MODEL_CONFIG, LORA_CONFIG


def load_model():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        device_map="cuda",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

    print("\n======== Trainable Parameters ========")
    model.print_trainable_parameters()
    print()

    return model, tokenizer