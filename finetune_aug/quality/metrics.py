# Computes automatic quality metrics (e.g., semantic similarity, SELF-BLEU)
# finetune_aug/quality/metrics.py
from finetune_aug.logger import setup_logger
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu

logger = setup_logger("metrics")
nltk.download('punkt', quiet=True)

def compute_self_bleu(augmented_examples):
    """
    Compute a basic self-BLEU score for the augmented outputs to measure diversity.
    """
    scores = []
    for i, ex in enumerate(augmented_examples):
        candidate = nltk.word_tokenize(ex.get("output", ""))
        references = [nltk.word_tokenize(other.get("output", "")) 
                      for j, other in enumerate(augmented_examples) if i != j]
        if references:
            score = sentence_bleu(references, candidate)
            scores.append(score)
    avg_bleu = np.mean(scores) if scores else 0
    logger.info("Average Self-BLEU: %.4f", avg_bleu)
    return avg_bleu

if __name__ == "__main__":
    test_augmented = [
        {"input": "what is the capital of france?", "output": "paris is the capital."},
        {"input": "what is the capital of france?", "output": "the capital city of france is paris."}
    ]
    bleu_score = compute_self_bleu(test_augmented)
    print("Self-BLEU score:", bleu_score)
