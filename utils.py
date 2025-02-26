"""
utils.py

This module provides a robust, scalable pipeline for finetuning data augmentation.
It includes data models, preprocessing functions, dynamic strategy selection,
augmentation generation via LangChain Groq, quality assurance, and post‐processing
to produce outputs in the OpenAI or Gemini fine‐tuning JSONL format.
"""

import os
import json
import uuid
import logging
import re
import random
import ast
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinetuningAugmentor")

# Environment tokens
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Must be set in .env

# -----------------------------
# Data Models and Preprocessing
# -----------------------------
from pydantic import BaseModel, field_validator, ValidationError

class AugmentationExample(BaseModel):
    """
    Input/output example for augmentation.
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
    system_message: Optional[str] = "Marv is a factual chatbot that is also sarcastic."

    @field_validator('examples')
    def check_examples_length(cls, v: List[AugmentationExample]) -> List[AugmentationExample]:
        if not (3 <= len(v) <= 10):
            raise ValueError("Provide between 3 and 10 examples")
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
    Validates that the record follows the required format for OpenAI:
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
    If it is a list, return the first element (recursively).
    If it is a dict and contains a "text" key, return that.
    If it is a string that looks like a dict, attempt to parse it.
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

# -----------------------------
# Augmentation Generation via LangChain Groq
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def instantiate_groq_llm(model: str) -> ChatGroq:
    """
    Instantiate a ChatGroq LLM with the given model name.
    """
    return ChatGroq(
        model=model,
        temperature=0.7,
        max_tokens=256,
        timeout=30,
        max_retries=2,
        groq_api_key=GROQ_API_KEY
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
    chain = prompt_template | instantiate_groq_llm(config.target_model)
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
    chain = refinement_template | instantiate_groq_llm(config.target_model)
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

def metrics_valid(metrics: dict) -> bool:
    """
    Validate metric thresholds:
      - Semantic similarity: between 0.80 and 0.95.
      - Diversity score: at least 0.70.
      - Fluency score: at least 0.80.
    """
    if metrics["semantic_similarity"] < 0.80 or metrics["semantic_similarity"] > 0.95:
        return False
    if metrics["diversity_score"] < 0.70:
        return False
    if metrics["fluency_score"] < 0.80:
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
                if not metrics_valid(metrics):
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
    Each record follows:
    {"messages": [{"role": "system", "content": system_message},
                   {"role": "user", "content": <augmented_input>},
                   {"role": "assistant", "content": <augmented_output>}]}
    """
    output_lines = []
    sys_msg = system_message.strip() if system_message and system_message.strip() else ""
    for aug in augmentations:
        user_val = get_text(aug["augmentation"].get("augmented_input", "")).strip()
        assistant_val = get_text(aug["augmentation"].get("augmented_output", "")).strip()
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
    Format augmentations in Gemini finetuning JSONL format.
    Each record follows:
    {
       "contents": [
          {"role": "user", "parts": [{"text": <augmented_input>}]},
          {"role": "model", "parts": [{"text": <augmented_output>}]}
       ]
    }
    """
    output_lines = []
    for aug in augmentations:
        user_val = get_text(aug["augmentation"].get("augmented_input", "")).strip()
        assistant_val = get_text(aug["augmentation"].get("augmented_output", "")).strip()
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

# -----------------------------
# Optional Interactive Review (Streamlit)
# -----------------------------
def launch_review_app(formatted_data: str):
    """
    Launch a Streamlit interface for interactive review of augmented data.
    """
    import streamlit as st
    st.title("Finetuning Data Augmentation Review")
    st.write("Review and, if necessary, edit the augmented data below.")
    edited_data = st.text_area("Augmented Data (JSONL)", formatted_data, height=400)
    if st.button("Approve and Save"):
        with open("train.jsonl", "w") as f:
            f.write(edited_data)
        st.success("Data saved successfully to train.jsonl!")

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
        For 'openai', include system message; for 'gemini', use Gemini format.
        """
        if format_type.lower() == "openai":
            return format_for_openai(self.augmentations, self.config.system_message)
        elif format_type.lower() == "gemini":
            return format_for_gemini(self.augmentations)
        else:
            logger.error(f"Unknown format type: {format_type}. Defaulting to OpenAI format.")
            return format_for_openai(self.augmentations, self.config.system_message)

    def save_to_file(self, filename: str = "train.jsonl"):
        """
        Save the formatted augmented data to a file.
        """
        output = self.get_formatted_output()
        with open(filename, "w") as f:
            f.write(output)
        logger.info(f"Final augmented data saved to {filename}")

    def run_review_interface(self):
        """
        Launch the interactive review interface.
        """
        formatted_data = self.get_formatted_output()
        launch_review_app(formatted_data)
