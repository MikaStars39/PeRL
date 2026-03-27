from .openr1 import load_openr1_dataset
from .tinyzero import load_tinyzero_dataset
from .count_down import load_count_down_dataset
from .still import load_still_dataset

from transformers import AutoTokenizer

def load_dataset(dataset_name_or_path: str, example_numbers: int = None, tokenizer: AutoTokenizer = None):
    dataset_name_lower = dataset_name_or_path.lower()
    if "r1" in dataset_name_lower:
        return load_openr1_dataset(dataset_name_or_path, example_numbers)
    elif "tinyzero" in dataset_name_lower:
        return load_tinyzero_dataset(dataset_name_or_path, example_numbers)
    elif "countdown" in dataset_name_lower:
        return load_count_down_dataset(
            dataset_name_or_path, 
            example_numbers, 
            tokenizer
        )
    elif "still" in dataset_name_lower:
        return load_still_dataset(dataset_name_or_path, example_numbers)
    else:
        raise ValueError(f"Not supported dataset: {dataset_name_or_path}")