import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

from config import DATA_PATH, TEST_CONFIG, MODEL_CONFIG


# Loading pretrained model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=DATA_PATH["model_dir"],
    max_seq_length=MODEL_CONFIG["max_seq_length"],
    dtype=None,
    load_in_4bit=MODEL_CONFIG["load_in_4bit"],
)

FastLanguageModel.for_inference(model)

tokenizer.pad_token = tokenizer.eos_token

# Prompting
def generate(question:str):

    prompt = f"""
### Instruction:
{question}

### Response:
"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(
        [prompt],
        return_tensors="pt"
    ).to(device)

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=TEST_CONFIG["max_new_tokens"],
            temperature=TEST_CONFIG["temperature"],
            top_p=TEST_CONFIG["top_p"],
            repetition_penalty=TEST_CONFIG["repetition_penalty"],
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# quick test
if __name__ == "__main__":
    generate("Who is Hadi?")