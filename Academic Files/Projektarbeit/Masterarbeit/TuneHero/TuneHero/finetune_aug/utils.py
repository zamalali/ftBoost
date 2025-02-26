# Shared helper functions and utilities
# finetune_aug/utils.py
import json

def load_examples(file_path: str):
    """
    Load user-provided examples from a text file.
    Each line should be in the format "input|output".
    """
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                inp, outp = line.strip().split("|", 1)
                examples.append({"input": inp.strip(), "output": outp.strip()})
    return examples

def save_jsonl(data, file_path: str):
    """
    Save a list of dictionaries to a JSONL file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
