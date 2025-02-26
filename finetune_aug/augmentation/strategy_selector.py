# Dynamically selects augmentation methods based on input and goal
# finetune_aug/augmentation/strategy_selector.py
from finetune_aug.logger import setup_logger

logger = setup_logger("strategy_selector")

def select_strategies(examples, goal: str):
    """
    Dynamically select augmentation strategies based on the input examples and finetuning goal.
    Returns a list of strategy identifiers.
    """
    strategies = []
    if "diversity" in goal.lower():
        strategies.extend(["synonym_replacement", "synthetic_noise"])
    if "context" in goal.lower() or "robust" in goal.lower():
        strategies.extend(["llm_paraphrasing", "back_translation"])
    strategies = list(set(strategies))
    logger.info("Selected strategies: %s", strategies)
    return strategies

if __name__ == "__main__":
    test_examples = [{"input": "What is the capital of France?", "output": "Paris"}]
    test_goal = "Increase diversity while preserving context"
    selected = select_strategies(test_examples, test_goal)
    print(selected)
