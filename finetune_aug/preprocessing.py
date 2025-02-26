# Preprocess and normalize user input data
# finetune_aug/preprocessing.py
from .logger import setup_logger
from .utils import load_examples

logger = setup_logger("preprocessing")

def preprocess_examples(file_path: str):
    """
    Load and preprocess user examples.
    For now, this function simply loads the examples and normalizes them by lowercasing.
    """
    logger.info("Loading examples from %s", file_path)
    examples = load_examples(file_path)
    for ex in examples:
        ex["input"] = ex["input"].lower().strip()
        ex["output"] = ex["output"].lower().strip()
    logger.info("Loaded %d examples.", len(examples))
    return examples

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Please provide the input file path")
    else:
        processed = preprocess_examples(sys.argv[1])
        print(processed)
