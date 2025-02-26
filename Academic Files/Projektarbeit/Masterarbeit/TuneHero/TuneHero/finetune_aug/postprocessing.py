# Deduplication, consistency checks, and formatting to model-specific output (e.g., JSONL)
# finetune_aug/postprocessing.py
import json
from finetune_aug.logger import setup_logger

logger = setup_logger("postprocessing")

def deduplicate_examples(augmented_data: list) -> list:
    """
    Remove duplicate examples based on identical input and output.
    """
    seen = set()
    unique = []
    for ex in augmented_data:
        key = (ex.get("input", "").strip(), ex.get("output", "").strip())
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    logger.info("Reduced examples from %d to %d after deduplication.", len(augmented_data), len(unique))
    return unique

def format_to_jsonl(augmented_data: list, output_file: str):
    """
    Format and save the augmented data to a JSONL file.
    """
    unique_data = deduplicate_examples(augmented_data)
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in unique_data:
            f.write(json.dumps(ex) + "\n")
    logger.info("Saved %d examples to %s", len(unique_data), output_file)

if __name__ == "__main__":
    sample_data = [
        {"input": "what is the capital of france?", "output": "paris is the capital."},
        {"input": "what is the capital of france?", "output": "paris is the capital."},
        {"input": "what is the capital of france?", "output": "the capital city of france is paris."}
    ]
    format_to_jsonl(sample_data, "augmented_data.jsonl")
