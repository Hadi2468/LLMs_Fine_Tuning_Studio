import torch

from transformers import TextStreamer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./models"
)

FastLanguageModel.for_inference(model)

question = "Who is Hadi?"

prompt = f"""
### Instruction:
{question}

### Response:
"""

inputs = tokenizer(
    [prompt],
    return_tensors="pt"
).to("cuda")

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True
)

with torch.inference_mode():

    model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=100
    )