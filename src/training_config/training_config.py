from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str
    max_seq_length: int
    load_in_4bit: bool

    r: int
    lora_alpha: int
    lora_dropout: float

    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    optim: str

    dataset_path: str | None = None