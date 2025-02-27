"""
cli.py

A command-line interface to run the augmentation pipeline.
Usage example:
    python -m finetune_augmentor.cli --file_path data/train.jsonl --format gemini --target_count 20
If --file_path is provided, the script loads existing examples from that file and then appends 
the new augmented pairs to the existing train.jsonl, preserving the original data.
If no file is provided, it uses hard-coded examples.
"""

import argparse
import os
from finetune_augmentor import (
    AugmentationConfig,
    FinetuningDataAugmentor,
    load_examples_from_file,
    AugmentationExample
)

def main():
    parser = argparse.ArgumentParser(description="Run finetuning data augmentation.")
    parser.add_argument("--format", type=str, default="openai", choices=["openai", "gemini"],
                        help="Output finetuning format")
    parser.add_argument("--target_count", type=int, default=50,
                        help="Number of augmented pairs to generate")
    parser.add_argument("--file_path", type=str, default="",
                        help="Path to an existing train.jsonl file with input/output pairs")
    parser.add_argument("--finetuning_goal", type=str, default="Enhance Q&A robustness and nuance preservation",
                        help="Finetuning goal description")
    # Optional thresholds
    parser.add_argument("--min_semantic_similarity", type=float, default=0.80)
    parser.add_argument("--max_semantic_similarity", type=float, default=0.95)
    parser.add_argument("--min_diversity_score", type=float, default=0.70)
    parser.add_argument("--min_fluency_score", type=float, default=0.80)
    parser.add_argument("--system_message", type=str, default="",
                        help="System message to use (for OpenAI format)")
    args = parser.parse_args()

    # Load existing examples from file if provided; otherwise, use hard-coded examples.
    if args.file_path and os.path.exists(args.file_path):
        examples = load_examples_from_file(args.file_path, format_type=args.format)
    else:
        examples = [
            AugmentationExample(input_text="What's the capital of France?", output_text="Paris"),
            AugmentationExample(input_text="Who wrote 'Romeo and Juliet'?", output_text="William Shakespeare"),
            AugmentationExample(input_text="How far is the Moon from Earth?", output_text="Approximately 384,400 kilometers")
        ]
    
    config = AugmentationConfig(
        target_model="mixtral-8x7b-32768",
        examples=examples,
        finetuning_goal=args.finetuning_goal,
        system_message=args.system_message,
        min_semantic_similarity=args.min_semantic_similarity,
        max_semantic_similarity=args.max_semantic_similarity,
        min_diversity_score=args.min_diversity_score,
        min_fluency_score=args.min_fluency_score
    )
    
    augmentor = FinetuningDataAugmentor(config)
    augmentor.run_augmentation(target_count=args.target_count)
    output_data = augmentor.get_formatted_output(format_type=args.format)
    
    # If the file exists, read existing content and append new data.
    output_file = "train.jsonl"
    existing_content = ""
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_content = f.read().strip()
    
    with open(output_file, "w") as f:
        if existing_content:
            f.write(existing_content + "\n")
        f.write(output_data)
    
    print(f"Augmented data appended to {output_file}")

if __name__ == "__main__":
    main()
