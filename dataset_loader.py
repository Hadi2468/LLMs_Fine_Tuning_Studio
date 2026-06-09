import json
import pandas as pd
from datasets import Dataset

def load_dataset(json_file):

    with open(json_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    return Dataset.from_pandas(df)