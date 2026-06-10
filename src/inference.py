import torch
import streamlit as st

from src.config import DATA_PATH, TEST_CONFIG, MODEL_CONFIG


# Loading pretrained model
@st.cache_resource
def load_finetuned_model():

    # lazy import
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(DATA_PATH["model_dir"]),
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        device_map="cuda",
    )

    FastLanguageModel.for_inference(model)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# Prompting
def generate(question:str, temperature=None, top_p=None):

    model, tokenizer = load_finetuned_model()

    # Prompt
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

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=TEST_CONFIG["max_new_tokens"],
            temperature=temperature or TEST_CONFIG["temperature"],
            top_p=top_p or TEST_CONFIG["top_p"],
            repetition_penalty=TEST_CONFIG["repetition_penalty"],
            pad_token_id=tokenizer.eos_token_id,
        )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if "### Response:" in decoded:
        decoded = decoded.split("### Response:")[-1].strip()
    
    return decoded


# quick test
if __name__ == "__main__":
    response = generate("Who is Sotude?")
    print(response)