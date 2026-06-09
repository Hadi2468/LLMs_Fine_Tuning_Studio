import json
import pandas as pd
from datasets import Dataset



# Convert JSON to DataFrame and then to Hugging Face Dataset
def load_dataset(json_file):

    with open(json_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    return Dataset.from_pandas(df)


# Format dataset for training
def format_dataset(dataset, tokenizer):

    def format_instruction(example):
        
        return {
            "text": 
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['response']}"
        }
    
    return dataset.map(format_instruction)
