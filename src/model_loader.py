def load_model(
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        ):

    import unsloth
    from unsloth import FastLanguageModel
    import torch

    device_map="cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"

    print("\n======== Trainable Parameters ========")
    model.print_trainable_parameters()
    print()

    return model, tokenizer
