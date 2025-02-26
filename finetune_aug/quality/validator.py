# Uses smolagents (with Hugging Face token) for alignment checks
# finetune_aug/quality/validator.py
from finetune_aug.config import HUGGINGFACE_TOKEN
from finetune_aug.logger import setup_logger

logger = setup_logger("validator")

# Hypothetical import for smolagents (ensure the library is installed)
try:
    from smolagents import Agent
except ImportError:
    logger.error("smolagents is not installed. Please install it to proceed.")
    Agent = None

def validate_example(example: dict, goal: str) -> bool:
    """
    Validate a single augmented example against the finetuning goal.
    Returns True if the example aligns well, otherwise False.
    """
    if Agent is None:
        raise ImportError("smolagents library is not available.")
    
    evaluation_prompt = (
        f"Evaluate the following augmentation example for alignment with the finetuning goal.\n"
        f"Goal: '{goal}'\n"
        f"Input: '{example.get('input')}'\n"
        f"Output: '{example.get('output')}'\n"
        f"Answer 'Yes' if it aligns well; otherwise, answer 'No'."
    )
    
    agent = Agent(prompt=evaluation_prompt, token=HUGGINGFACE_TOKEN)
    result = agent.run().strip().lower()
    logger.info("Validation result: %s", result)
    return "yes" in result

if __name__ == "__main__":
    test_example = {"input": "what is the capital of france?", "output": "paris is the capital of france."}
    test_goal = "Enhance Q&A robustness by diversifying paraphrases"
    is_valid = validate_example(test_example, test_goal)
    print("Validation passed:", is_valid)
