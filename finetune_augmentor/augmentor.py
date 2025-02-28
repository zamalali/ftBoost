"""
augmentor.py

This module implements a robust and scalable pipeline for finetuning data augmentation.
It supports generating augmented data in either OpenAI, Gemini, Mistral, or LLama fine‐tuning JSONL format.
Users may optionally override metric thresholds and load existing examples from a JSONL file.
The LangChain Groq API key is now provided via the configuration rather than the .env file.
"""

import os
import json
import uuid
import logging
import re
import random
import ast
from typing import List, Dict, Any, Optional

# Removed dotenv load for GROQ_API_KEY since it is now provided in config
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinetuningAugmentor")

# Environment tokens (kept for HF_TOKEN if needed)
HF_TOKEN = os.getenv("HF_TOKEN")
# GROQ_API_KEY will now be provided in the configuration

# -----------------------------
# Data Models and Preprocessing
# -----------------------------
from pydantic import BaseModel, field_validator, ValidationError

class AugmentationExample(BaseModel):
    """
    An input/output example for augmentation.
    """
    input_text: str
    output_text: str

    @field_validator('input_text', 'output_text')
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text fields must be non-empty")
        return v.strip()

class AugmentationConfig(BaseModel):
    """
    Configuration for the augmentation process.
    """
    target_model: str  # e.g., "mixtral-8x7b-32768" or any Groq-supported model name
    examples: List[AugmentationExample]
    finetuning_goal: str
    groq_api_key: str
    system_message: Optional[str] = "Marv is a factual chatbot that is also sarcastic."
    # Optional metric thresholds (if not provided, defaults are used)
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
    """
    Standardized format for input examples.
    """
    id: str
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = {}

def normalize_examples(examples: List[AugmentationExample]) -> List[StandardExample]:
    """
    Normalize and standardize input examples.
    """
    normalized = []
    for ex in examples:
        norm_ex = StandardExample(
            id=str(uuid.uuid4()),
            input_text=ex.input_text.lower(),
            output_text=ex.output_text.lower(),
            metadata={"original_word_count": len(ex.input_text.split())}
        )
        normalized.append(norm_ex)
    logger.info(f"Normalized {len(normalized)} examples.")
    return normalized

# -----------------------------
# Dynamic Strategy Selection
# -----------------------------
def determine_augmentation_strategy(config: AugmentationConfig) -> Dict[str, Any]:
    """
    Determine the augmentation strategy based on the finetuning goal.
    """
    goal = config.finetuning_goal.lower()
    strategy = {}
    if any(word in goal for word in ["dialogue", "q&a", "conversation", "chat"]):
        strategy["methods"] = ["llm_paraphrasing", "back_translation"]
    else:
        strategy["methods"] = ["eda_synonym_replacement", "llm_paraphrasing", "synthetic_noise"]
    strategy["diversity_threshold"] = 0.7
    logger.info(f"Determined augmentation strategy: {strategy}")
    return strategy

# -----------------------------
# Helper Functions
# -----------------------------
def extract_json(text: str) -> dict:
    """
    Extract the first valid JSON object from a given text.
    """
    json_pattern = re.compile(r'\{.*\}', re.DOTALL)
    match = json_pattern.search(text)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error: {e}")
    else:
        raise ValueError("No valid JSON found in text.")

def make_hashable(item: Any) -> Any:
    """
    Recursively convert unhashable types (lists/dicts) into hashable tuples.
    """
    if isinstance(item, (list, tuple)):
        return tuple(make_hashable(i) for i in item)
    elif isinstance(item, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    else:
        return item

def validate_jsonl_record(record: dict) -> bool:
    """
    Validates that the record follows the required OpenAI format:
    {"messages": [{"role": "system", "content": <str>},
                  {"role": "user", "content": <non-empty str>},
                  {"role": "assistant", "content": <non-empty str>}]}
    """
    if "messages" not in record:
        logger.error("Record missing 'messages' key.")
        return False
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        logger.error("Record 'messages' must be a list of 3 items.")
        return False
    expected_roles = ["system", "user", "assistant"]
    for msg, role in zip(messages, expected_roles):
        if not isinstance(msg, dict):
            logger.error("Each message must be a dictionary.")
            return False
        if msg.get("role") != role:
            logger.error(f"Expected role '{role}', but got '{msg.get('role')}'.")
            return False
        if "content" not in msg or not isinstance(msg["content"], str):
            logger.error("Each message must have a string 'content' field.")
            return False
        if role in ["user", "assistant"] and not msg["content"].strip():
            logger.error(f"Message for role '{role}' has empty content.")
            return False
    return True

def get_text(value: Any) -> str:
    """
    Ensure the value is returned as a string.
    If it is a list, recursively return the first element.
    If it is a dict and contains a "text" key, return that.
    If it is a string that resembles a dict, try to parse it.
    """
    if isinstance(value, list):
        if value:
            return get_text(value[0])
        return ""
    elif isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        return str(value)
    elif isinstance(value, str):
        val = value.strip()
        if val.startswith("{") and val.endswith("}"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, dict) and "text" in parsed:
                    return str(parsed["text"])
            except Exception:
                pass
        return val
    else:
        return str(value)

# --- New helper: Fix content formatting ---
def fix_content(content: str) -> str:
    """
    If the content appears to be a Python dict (using single quotes), try to
    convert it to valid JSON (with double quotes). If parsing fails, return the original content.
    """
    content = content.strip()
    if content.startswith("{") and content.endswith("}") and "'" in content:
        try:
            parsed = ast.literal_eval(content)
            return json.dumps(parsed)
        except Exception as e:
            logger.debug(f"Failed to fix content formatting: {e}")
    return content

def flatten_content(content: str) -> str:
    """
    If content (after fixing) is a JSON string representing a dictionary,
    flatten it by joining its values into a single plain-text string.
    """
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # Join values in sorted order by key
            values = [str(parsed[k]).strip() for k in sorted(parsed)]
            return " ".join(values)
    except Exception:
        pass
    return content

# -----------------------------
# Augmentation Generation via LangChain Groq
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def instantiate_groq_llm(model: str, groq_api_key: str) -> ChatGroq:
    """
    Instantiate a ChatGroq LLM with the given model name and API key.
    """
    return ChatGroq(
        model=model,
        temperature=0.7,
        max_tokens=256,
        timeout=30,
        max_retries=2,
        groq_api_key=groq_api_key
    )

def generate_initial_augmentation(example: StandardExample,
                                    config: AugmentationConfig,
                                    strategy: Dict[str, Any]) -> dict:
    """
    Generate an initial candidate augmentation using an LLM prompt chain.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            ("You are a creative augmentation assistant that produces diverse yet semantically consistent "
             "input/output pairs for finetuning tasks.")
        ),
        (
            "human",
            (
                "Augment the following example using the methods: {methods}. The finetuning goal is: {finetuning_goal}.\n"
                "Ensure your output is in valid JSON format with keys 'augmented_input' and 'augmented_output'.\n"
                "Input: {input_text}\n"
                "Output: {output_text}\n"
                "Return only the JSON response."
            )
        )
    ])
    prompt_vars = {
        "methods": ", ".join(strategy["methods"]),
        "finetuning_goal": config.finetuning_goal,
        "input_text": example.input_text,
        "output_text": example.output_text
    }
    chain = prompt_template | instantiate_groq_llm(config.target_model, config.groq_api_key)
    ai_msg = chain.invoke(prompt_vars)
    logger.info(f"Initial augmentation for {example.id}: {ai_msg.content.strip()}")
    return extract_json(ai_msg.content.strip())

def refine_augmentation(candidate: dict,
                        example: StandardExample,
                        config: AugmentationConfig,
                        strategy: Dict[str, Any]) -> dict:
    """
    Refine a candidate augmentation using a second LLM prompt chain.
    """
    refinement_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert data augmentation advisor who refines candidate augmentations to maximize semantic accuracy, diversity, and clarity."
        ),
        (
            "human",
            (
                "Review the candidate augmentation for the following input/output pair and refine it if needed.\n"
                "Finetuning Goal: {finetuning_goal}\n"
                "Original Input: {input_text}\n"
                "Original Output: {output_text}\n"
                "Candidate Augmentation: {candidate}\n"
                "Return a refined augmentation in valid JSON format with keys 'augmented_input' and 'augmented_output' only."
            )
        )
    ])
    refinement_vars = {
        "finetuning_goal": config.finetuning_goal,
        "input_text": example.input_text,
        "output_text": example.output_text,
        "candidate": json.dumps(candidate)
    }
    chain = refinement_template | instantiate_groq_llm(config.target_model, config.groq_api_key)
    ai_msg = chain.invoke(refinement_vars)
    try:
        refined = extract_json(ai_msg.content.strip())
        logger.info(f"Refined augmentation for {example.id}: {refined}")
        return refined
    except Exception as e:
        logger.error(f"Refinement failed for {example.id}: {e}. Using original candidate.")
        return candidate

def calculate_metrics(augmentation: dict, original: StandardExample) -> dict:
    """
    Simulate metric calculations for the candidate augmentation.
    """
    semantic_similarity = random.uniform(0.78, 0.97)
    diversity_score = random.uniform(0.65, 0.9)
    fluency_score = random.uniform(0.80, 0.95)
    metrics = {
        "semantic_similarity": semantic_similarity,
        "diversity_score": diversity_score,
        "fluency_score": fluency_score
    }
    logger.info(f"Metrics for candidate of {original.id}: {metrics}")
    return metrics

def metrics_valid(metrics: dict, config: AugmentationConfig) -> bool:
    """
    Validate metric thresholds using configuration values.
    """
    if metrics["semantic_similarity"] < config.min_semantic_similarity or metrics["semantic_similarity"] > config.max_semantic_similarity:
        return False
    if metrics["diversity_score"] < config.min_diversity_score:
        return False
    if metrics["fluency_score"] < config.min_fluency_score:
        return False
    return True

def quality_check(augmentation: Dict[str, Any], config: AugmentationConfig) -> bool:
    """
    Simulate an LLM-based QA check.
    """
    qa_prompt = (
        f"Verify that the following augmentation preserves the intended meaning and style for the input/output pair "
        f"given the finetuning goal '{config.finetuning_goal}':\n"
        f"{augmentation['augmentation']}\n"
        "Answer 'yes' if valid, otherwise 'no'."
    )
    logger.debug(f"QA Prompt: {qa_prompt}")
    return True  # Simulation: always passes

def generate_augmentations(normalized_examples: List[StandardExample],
                           config: AugmentationConfig,
                           strategy: Dict[str, Any],
                           target_count: int = 50) -> List[Dict[str, Any]]:
    """
    Repeatedly generate candidate augmentations until at least target_count valid candidates are collected.
    """
    augmented_candidates = []
    attempts = 0
    max_attempts = 100  # Safety valve
    while len(augmented_candidates) < target_count and attempts < max_attempts:
        for ex in normalized_examples:
            try:
                candidate = generate_initial_augmentation(ex, config, strategy)
                refined_candidate = refine_augmentation(candidate, ex, config, strategy)
                metrics = calculate_metrics(refined_candidate, ex)
                if not metrics_valid(metrics, config):
                    logger.info(f"Candidate for {ex.id} rejected by metrics: {metrics}")
                    continue
                if quality_check({"augmentation": refined_candidate}, config):
                    full_candidate = {
                        "original_id": ex.id,
                        "augmentation": refined_candidate,
                        "strategy": strategy,
                        "metrics": metrics
                    }
                    augmented_candidates.append(full_candidate)
                    logger.info(f"Accepted candidate for {ex.id} (Total accepted: {len(augmented_candidates)})")
                    if len(augmented_candidates) >= target_count:
                        break
            except Exception as e:
                logger.error(f"Error generating augmentation for {ex.id}: {e}")
        attempts += 1
    if len(augmented_candidates) < target_count:
        logger.warning(f"Only {len(augmented_candidates)} candidates generated after {attempts} attempts.")
    return augmented_candidates

def deduplicate_augmentations(augmentations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate augmentations based on hashable keys.
    """
    seen = set()
    unique_aug = []
    for aug in augmentations:
        key = (make_hashable(aug["augmentation"].get("augmented_input")),
               make_hashable(aug["augmentation"].get("augmented_output")))
        if key not in seen:
            seen.add(key)
            unique_aug.append(aug)
    logger.info(f"Deduplicated to {len(unique_aug)} unique augmentations.")
    return unique_aug

def format_for_openai(augmentations: List[Dict[str, Any]], system_message: str) -> str:
    """
    Format augmentations in OpenAI fine-tuning JSONL format.
    """
    output_lines = []
    sys_msg = system_message.strip() if system_message and system_message.strip() else ""
    for aug in augmentations:
        user_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_input", "")).strip()))
        assistant_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_output", "")).strip()))
        record = {
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_val},
                {"role": "assistant", "content": assistant_val}
            ]
        }
        if validate_jsonl_record(record):
            output_lines.append(json.dumps(record))
        else:
            logger.error(f"Record validation failed: {record}")
    logger.info(f"Formatted {len(output_lines)} records in OpenAI fine-tuning format.")
    return "\n".join(output_lines)

def format_for_gemini(augmentations: List[Dict[str, Any]]) -> str:
    """
    Format augmentations in Gemini fine-tuning JSONL format.
    """
    output_lines = []
    for aug in augmentations:
        user_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_input", "")).strip()))
        assistant_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_output", "")).strip()))
        record = {
            "contents": [
                {"role": "user", "parts": [{"text": user_val}]},
                {"role": "model", "parts": [{"text": assistant_val}]}
            ]
        }
        if user_val and assistant_val:
            output_lines.append(json.dumps(record))
        else:
            logger.error(f"Gemini record validation failed: {record}")
    logger.info(f"Formatted {len(output_lines)} records in Gemini fine-tuning format.")
    return "\n".join(output_lines)

def format_for_common(augmentations: List[Dict[str, Any]]) -> str:
    """
    Format augmentations in a common JSONL format for both Mistral and LLama.
    """
    output_lines = []
    for aug in augmentations:
        user_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_input", "")).strip()))
        assistant_val = flatten_content(fix_content(get_text(aug["augmentation"].get("augmented_output", "")).strip()))
        record = {
            "messages": [
                {"role": "user", "content": user_val},
                {"role": "assistant", "content": assistant_val}
            ]
        }
        if user_val and assistant_val:
            output_lines.append(json.dumps(record))
        else:
            logger.error(f"Common format record validation failed: {record}")
    logger.info(f"Formatted {len(output_lines)} records in common JSONL format for Mistral/LLama.")
    return "\n".join(output_lines)

def format_for_mistral(augmentations: List[Dict[str, Any]]) -> str:
    """
    Format augmentations in Mistral fine-tuning JSONL format.
    Uses the common format.
    """
    return format_for_common(augmentations)

def format_for_llama(augmentations: List[Dict[str, Any]]) -> str:
    """
    Format augmentations in LLama fine-tuning JSONL format.
    Uses the common format.
    """
    return format_for_common(augmentations)

# -----------------------------
# Optional: Load Existing Examples from JSONL
# -----------------------------
def load_examples_from_file(file_path: str, format_type: str = "openai") -> List[AugmentationExample]:
    """
    Load input/output pairs from a JSONL file.
    """
    examples = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if format_type.lower() == "openai":
                    msgs = record.get("messages", [])
                    if len(msgs) == 3:
                        user_text = msgs[1].get("content", "").strip()
                        assistant_text = msgs[2].get("content", "").strip()
                        if user_text and assistant_text:
                            examples.append(AugmentationExample(input_text=user_text, output_text=assistant_text))
                elif format_type.lower() == "gemini":
                    contents = record.get("contents", [])
                    if len(contents) >= 2:
                        user_parts = contents[0].get("parts", [])
                        model_parts = contents[1].get("parts", [])
                        user_text = get_text(user_parts[0]) if user_parts else ""
                        assistant_text = get_text(model_parts[0]) if model_parts else ""
                        if user_text and assistant_text:
                            examples.append(AugmentationExample(input_text=user_text, output_text=assistant_text))
    except Exception as e:
        logger.error(f"Error loading examples from file: {e}")
    logger.info(f"Loaded {len(examples)} examples from {file_path}")
    return examples

# -----------------------------
# Pipeline Class
# -----------------------------
class FinetuningDataAugmentor:
    """
    Encapsulates the entire augmentation pipeline.
    """
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.normalized_examples = normalize_examples(config.examples)
        self.strategy = determine_augmentation_strategy(config)
        self.augmentations = []

    def run_augmentation(self, target_count: int = 50) -> List[Dict[str, Any]]:
        """
        Generate candidate augmentations, deduplicate, and store results.
        """
        logger.info("Starting augmentation generation via LangChain Groq...")
        candidates = generate_augmentations(self.normalized_examples, self.config, self.strategy, target_count=target_count)
        logger.info(f"Generated {len(candidates)} candidate augmentations before deduplication.")
        unique_candidates = deduplicate_augmentations(candidates)
        logger.info(f"{len(unique_candidates)} unique augmentations after deduplication.")
        self.augmentations = unique_candidates
        return unique_candidates

    def get_formatted_output(self, format_type: str = "openai") -> str:
        """
        Return the final augmented data in the desired finetuning format.
        """
        fmt = format_type.lower()
        if fmt == "openai":
            return format_for_openai(self.augmentations, self.config.system_message)
        elif fmt == "gemini":
            return format_for_gemini(self.augmentations)
        elif fmt == "mistral":
            return format_for_mistral(self.augmentations)
        elif fmt == "llama":
            return format_for_llama(self.augmentations)
        else:
            logger.error(f"Unknown format type: {format_type}. Defaulting to OpenAI format.")
            return format_for_openai(self.augmentations, self.config.system_message)

    def save_to_file(self, filename: str = "train.jsonl") -> None:
        """
        Save the formatted augmented data to a file.
        """
        output = self.get_formatted_output()
        with open(filename, "w") as f:
            f.write(output)
        logger.info(f"Final augmented data saved to {filename}")

    def run_review_interface(self) -> None:
        """
        Launch the interactive review interface.
        """
        from streamlit import runtime
        formatted_data = self.get_formatted_output()
        launch_review_app(formatted_data)
