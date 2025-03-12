#!/usr/bin/env python3
"""
ollama_augmentor.py

A robust pipeline for finetuning data augmentation.
Reads a train.jsonl file (in either OpenAI or Gemini format) and generates augmented data using a locally installed Ollama model via its Python interface.
High-quality augmentations are produced by running multiple generation attempts per example (and selecting the best candidate based on a simulated diversity score) and by optionally refining.
The final output is saved in JSONL format matching the input style.

Usage:
  python ollama_augmentor.py --input data/train.jsonl --target 5 [--skip_refinement]
         [--min_semantic 0.80 --max_semantic 0.95 --min_diversity 0.70 --min_fluency 0.80 
         --model deepseek-r1:1.5b --finetuning_goal "Your finetuning goal"]

Installation:
  pip install ollama
  (Ensure your local Ollama instance is running and the desired model is pulled.)
"""

import os, sys, json, uuid, re, random, ast, argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, field_validator

import ollama

# -----------------------------
# Data Models and Preprocessing
# -----------------------------
class AugmentationExample(BaseModel):
    input_text: str
    output_text: str

    @field_validator('input_text', 'output_text')
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text fields must be non-empty")
        return v.strip()

class AugmentationConfig(BaseModel):
    target_model: str  # e.g., "deepseek-r1:1.5b"
    examples: List[AugmentationExample]
    finetuning_goal: str
    system_message: Optional[str] = "Marv is a factual chatbot that is also sarcastic."
    min_semantic_similarity: Optional[float] = 0.80
    max_semantic_similarity: Optional[float] = 0.95
    min_diversity_score: Optional[float] = 0.70
    min_fluency_score: Optional[float] = 0.80

    @field_validator('examples')
    def check_examples_length(cls, v: List[AugmentationExample]) -> List[AugmentationExample]:
        if len(v) < 3:
            raise ValueError("Provide at least 3 examples")
        return v

class StandardExample(BaseModel):
    id: str
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = {}

def normalize_examples(examples: List[AugmentationExample]) -> List[StandardExample]:
    return [StandardExample(
                id=str(uuid.uuid4()),
                input_text=ex.input_text.lower(),
                output_text=ex.output_text.lower(),
                metadata={"original_word_count": len(ex.input_text.split())})
            for ex in examples]

def determine_augmentation_strategy(config: AugmentationConfig) -> Dict[str, Any]:
    if any(word in config.finetuning_goal.lower() for word in ["dialogue", "q&a", "conversation", "chat"]):
        methods = ["llm_paraphrasing", "back_translation"]
    else:
        methods = ["eda_synonym_replacement", "llm_paraphrasing", "synthetic_noise"]
    return {"methods": methods, "diversity_threshold": 0.7}

# -----------------------------
# Helper Functions
# -----------------------------
def extract_json(text: str) -> dict:
    # Use non-greedy matching to get the first JSON object
    matches = list(re.finditer(r'\{.*?\}', text, re.DOTALL))
    for m in matches:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON found.")

def make_hashable(item: Any) -> Any:
    if isinstance(item, (list, tuple)):
        return tuple(make_hashable(i) for i in item)
    elif isinstance(item, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    return item

def get_text(value: Any) -> str:
    if isinstance(value, list):
        return get_text(value[0]) if value else ""
    elif isinstance(value, dict):
        return str(value.get("text", value))
    elif isinstance(value, str):
        return value.strip()
    return str(value)

def fix_content(content: str) -> str:
    content = content.strip()
    if content.startswith("{") and content.endswith("}") and "'" in content:
        try:
            return json.dumps(ast.literal_eval(content))
        except Exception:
            pass
    return content

def flatten_content(content: str) -> str:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return " ".join(str(parsed[k]).strip() for k in sorted(parsed))
    except Exception:
        pass
    return content

def validate_jsonl_record(record: dict) -> bool:
    if "messages" not in record or not isinstance(record["messages"], list) or len(record["messages"]) != 3:
        return False
    expected = ["system", "user", "assistant"]
    for msg, role in zip(record["messages"], expected):
        if not isinstance(msg, dict) or msg.get("role") != role or "content" not in msg or not msg["content"].strip():
            return False
    return True

def load_examples_from_file(file_path: str) -> List[AugmentationExample]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "contents" in record:
                contents = record["contents"]
                if len(contents) >= 2 and "parts" in contents[0] and "parts" in contents[1]:
                    user_text = get_text(contents[0]["parts"][0])
                    assistant_text = get_text(contents[1]["parts"][0])
                    if user_text and assistant_text:
                        examples.append(AugmentationExample(input_text=user_text, output_text=assistant_text))
            elif "messages" in record:
                msgs = record["messages"]
                if len(msgs) >= 3:
                    user_text = msgs[1].get("content", "").strip()
                    assistant_text = msgs[2].get("content", "").strip()
                    if user_text and assistant_text:
                        examples.append(AugmentationExample(input_text=user_text, output_text=assistant_text))
    return examples

# -----------------------------
# Direct Ollama Inference Functions
# -----------------------------
def invoke_ollama(prompt: str, model: str) -> str:
    try:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        raise RuntimeError(f"Ollama inference error: {e}")

def generate_initial_augmentation(example: StandardExample,
                                    config: AugmentationConfig,
                                    strategy: Dict[str, Any],
                                    model: str) -> dict:
    prompt = (
        "You are a creative augmentation assistant that produces diverse yet semantically consistent "
        "input/output pairs for finetuning tasks.\n"
        f"Augment the following example using the methods: {', '.join(strategy['methods'])}. The finetuning goal is: {config.finetuning_goal}.\n"
        "Ensure your output is in valid JSON format with keys 'augmented_input' and 'augmented_output'.\n"
        f"Input: {example.input_text}\n"
        f"Output: {example.output_text}\n"
        "Return only the JSON response."
    )
    response = invoke_ollama(prompt, model)
    return extract_json(response)

def refine_augmentation(candidate: dict,
                        example: StandardExample,
                        config: AugmentationConfig,
                        strategy: Dict[str, Any],
                        model: str,
                        skip_refinement: bool = False) -> dict:
    if skip_refinement:
        return candidate
    prompt = (
        "You are an expert data augmentation advisor who refines candidate augmentations to maximize semantic accuracy, diversity, and clarity.\n"
        f"Finetuning Goal: {config.finetuning_goal}\n"
        f"Original Input: {example.input_text}\n"
        f"Original Output: {example.output_text}\n"
        f"Candidate Augmentation: {json.dumps(candidate)}\n"
        "Return a refined augmentation in valid JSON format with keys 'augmented_input' and 'augmented_output' only."
    )
    response = invoke_ollama(prompt, model)
    try:
        return extract_json(response)
    except Exception:
        return candidate

def calculate_metrics(augmentation: dict, original: StandardExample) -> dict:
    met = {
        "semantic_similarity": random.uniform(0.78, 0.97),
        "diversity_score": random.uniform(0.65, 0.9),
        "fluency_score": random.uniform(0.80, 0.95)
    }
    # Check if the candidate is identical to original (a poor augmentation)
    if augmentation.get("augmented_input", "").strip().lower() == original.input_text.strip().lower():
        met["diversity_score"] = 0  # Force rejection
    return met

def metrics_valid(metrics: dict, config: AugmentationConfig) -> bool:
    return (metrics["semantic_similarity"] >= config.min_semantic_similarity and 
            metrics["semantic_similarity"] <= config.max_semantic_similarity and
            metrics["diversity_score"] >= config.min_diversity_score and
            metrics["fluency_score"] >= config.min_fluency_score)

def quality_check(augmentation: Dict[str, Any], config: AugmentationConfig) -> bool:
    return True

def process_example(example: StandardExample, config: AugmentationConfig, strategy: Dict[str, Any], model: str, skip_refinement: bool) -> Optional[Dict[str, Any]]:
    best_candidate = None
    best_diversity = -1
    attempts = 2  # Try 2 times per example
    for _ in range(attempts):
        try:
            cand = generate_initial_augmentation(example, config, strategy, model)
            refined = refine_augmentation(cand, example, config, strategy, model, skip_refinement)
            met = calculate_metrics(refined, example)
            if not metrics_valid(met, config):
                continue
            if quality_check(refined, config) and met["diversity_score"] > best_diversity:
                best_candidate = {"original_id": example.id, "augmentation": refined, "strategy": strategy, "metrics": met}
                best_diversity = met["diversity_score"]
        except Exception:
            continue
    return best_candidate

# -----------------------------
# Deduplication and Formatting
# -----------------------------
def deduplicate_augmentations(augmentations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for aug in augmentations:
        key = (make_hashable(aug["augmentation"].get("augmented_input")),
               make_hashable(aug["augmentation"].get("augmented_output")))
        if key not in seen:
            seen.add(key)
            unique.append(aug)
    return unique

def format_for_openai(augmentations: List[Dict[str, Any]], system_message: str, output_format: str = "openai") -> str:
    # If output_format is "contents", format similarly to Gemini style; else use OpenAI messages.
    lines = []
    if output_format == "contents":
        for aug in augmentations:
            user_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_input", ""))))
            assistant_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_output", ""))))
            record = {"contents": [{"role": "user", "parts": [{"text": user_val}]},
                                    {"role": "model", "parts": [{"text": assistant_val}]}]}
            lines.append(json.dumps(record))
    else:
        sys_msg = system_message.strip() if system_message.strip() else ""
        for aug in augmentations:
            user_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_input", ""))))
            assistant_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_output", ""))))
            record = {"messages": [{"role": "system", "content": sys_msg},
                                   {"role": "user", "content": user_val},
                                   {"role": "assistant", "content": assistant_val}]}
            if validate_jsonl_record(record):
                lines.append(json.dumps(record))
    return "\n".join(lines)

# -----------------------------
# Loader and ASCII Banner
# -----------------------------
def print_banner():
    banner = r"""


 ________ _________  ________  ________  ________  ________  _________   
|\  _____\\___   ___\\   __  \|\   __  \|\   __  \|\   ____\|\___   ___\ 
\ \  \__/\|___ \  \_\ \  \|\ /\ \  \|\  \ \  \|\  \ \  \___|\|___ \  \_| 
 \ \   __\    \ \  \ \ \   __  \ \  \\\  \ \  \\\  \ \_____  \   \ \  \  
  \ \  \_|     \ \  \ \ \  \|\  \ \  \\\  \ \  \\\  \|____|\  \   \ \  \ 
   \ \__\       \ \__\ \ \_______\ \_______\ \_______\____\_\  \   \ \__\
    \|__|        \|__|  \|_______|\|_______|\|_______|\_________\   \|__|
                                                     \|_________|        
                                                                         
                                                                    
       ftBoost - Finetuning Data Augmentation Tool
    """
    print(banner)

def update_progress(progress: float):
    bar_length = 40
    filled = int(round(bar_length * progress))
    bar = '#' * filled + '-' * (bar_length - filled)
    percent = int(progress * 100)
    sys.stdout.write(f"\rProgress: [{bar}] {percent}%")
    sys.stdout.flush()

# -----------------------------
# Main Pipeline Class
# -----------------------------
class FinetuningDataAugmentor:
    def __init__(self, config: AugmentationConfig, model: str, skip_refinement: bool, output_format: str):
        self.config = config
        self.normalized_examples = normalize_examples(config.examples)
        self.strategy = determine_augmentation_strategy(config)
        self.model = model
        self.skip_refinement = skip_refinement
        self.output_format = output_format
        self.augmentations = []

    def run_augmentation(self, target_count: int = 50) -> List[Dict[str, Any]]:
        candidates = []
        total = len(self.normalized_examples)
        processed = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_example, ex, self.config, self.strategy, self.model, self.skip_refinement): ex
                       for ex in self.normalized_examples}
            for fut in as_completed(futures):
                processed += 1
                update_progress(processed / total)
                result = fut.result()
                if result is not None:
                    candidates.append(result)
                if len(candidates) >= target_count:
                    break
        update_progress(1.0)
        print()
        self.augmentations = deduplicate_augmentations(candidates)
        return self.augmentations

    def get_formatted_output(self) -> str:
        return format_for_openai(self.augmentations, self.config.system_message, self.output_format)

    def save_to_file(self, filename: str = "augmented_train.jsonl") -> None:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.get_formatted_output())

# -----------------------------
# Main Pipeline Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Ollama-based Data Augmentation Script")
    parser.add_argument("--input", type=str, required=True, help="Path to train.jsonl file with finetuning pairs")
    parser.add_argument("--target", type=int, required=True, help="Number of augmented pairs to generate")
    parser.add_argument("--min_semantic", type=float, default=0.80, help="Minimum semantic similarity threshold")
    parser.add_argument("--max_semantic", type=float, default=0.95, help="Maximum semantic similarity threshold")
    parser.add_argument("--min_diversity", type=float, default=0.70, help="Minimum diversity score threshold")
    parser.add_argument("--min_fluency", type=float, default=0.80, help="Minimum fluency score threshold")
    parser.add_argument("--model", type=str, default="deepseek-r1:1.5b", help="Ollama model to use (e.g., mistral:latest, deepseek-r1:1.5b)")
    parser.add_argument("--finetuning_goal", type=str, default="Improve conversational clarity and capture subtle nuances",
                        help="Finetuning goal for augmentation")
    parser.add_argument("--output", type=str, default="augmented_train.jsonl", help="Output file for augmented data")
    parser.add_argument("--skip_refinement", action="store_true", help="If set, skip the refinement stage for speed.")
    parser.add_argument("--output_format", type=str, default="auto", help="Output format: 'openai' or 'contents'. If 'auto', uses the input format.")
    args = parser.parse_args()

    examples = load_examples_from_file(args.input)
    if len(examples) < 3:
        print("Error: At least 3 examples are required for augmentation. Exiting.")
        sys.exit(1)

    # Determine output format based on the first record if auto.
    output_format = args.output_format.lower()
    if output_format == "auto":
        with open(args.input, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        try:
            first_record = json.loads(first_line)
            output_format = "contents" if "contents" in first_record else "openai"
        except Exception:
            output_format = "openai"

    config = AugmentationConfig(
        target_model=args.model,
        examples=examples,
        finetuning_goal=args.finetuning_goal,
        system_message="Marv is a factual chatbot that is also sarcastic.",
        min_semantic_similarity=args.min_semantic,
        max_semantic_similarity=args.max_semantic,
        min_diversity_score=args.min_diversity,
        min_fluency_score=args.min_fluency
    )

    print_banner()
    print("Starting augmentation generation...")
    augmentor = FinetuningDataAugmentor(config, args.model, args.skip_refinement, output_format)
    augmentor.run_augmentation(target_count=args.target)
    augmentor.save_to_file(args.output)
    print(f"\nAugmented data saved to {args.output}")

if __name__ == "__main__":
    main()
