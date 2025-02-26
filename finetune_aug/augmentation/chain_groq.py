# Implements the langchain_groq interactive augmentation chain
# finetune_aug/augmentation/chain_groq.py
from finetune_aug.config import HUGGINGFACE_TOKEN, AUGMENTATION_TARGET_COUNT
from finetune_aug.logger import setup_logger

logger = setup_logger("chain_groq")

# Hypothetical import for langchain_groq (ensure you have the actual package installed)
try:
    from langchain_groq import GroqChain
except ImportError:
    logger.error("langchain_groq is not installed. Please install it to continue.")
    GroqChain = None

def build_augmentation_chain(examples, goal: str, target_count: int = AUGMENTATION_TARGET_COUNT):
    """
    Build an interactive augmentation chain using langchain_groq.
    Generates candidate augmented examples in JSONL format.
    """
    if GroqChain is None:
        raise ImportError("langchain_groq library is not available.")
    
    prompt_template = (
        "You are a data augmentation expert. The user provided the following examples:\n"
        "{examples}\n\n"
        "The finetuning goal is: '{goal}'.\n"
        "Generate {target_count} high-quality augmented input-output pairs that align with this goal.\n"
        "Return the output in JSONL format with keys 'input' and 'output'."
    )
    
    chain = GroqChain(
        prompt_template=prompt_template,
        input_variables=["examples", "goal", "target_count"],
        llm_kwargs={"temperature": 0.7, "api_token": HUGGINGFACE_TOKEN}
    )
    
    examples_str = "\n".join(
        [f"Input: {ex['input']} | Output: {ex['output']}" for ex in examples]
    )
    
    logger.info("Running augmentation chain...")
    augmented_output = chain.run({
        "examples": examples_str,
        "goal": goal,
        "target_count": target_count
    })
    logger.info("Augmentation chain completed.")
    return augmented_output

if __name__ == "__main__":
    test_examples = [{"input": "What is the capital of France?", "output": "Paris"}]
    test_goal = "Enhance robustness in Q&A style"
    try:
        output = build_augmentation_chain(test_examples, test_goal)
        print("Generated Augmentations:")
        print(output)
    except Exception as e:
        logger.error("Error during augmentation chain execution: %s", str(e))
