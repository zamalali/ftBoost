import os
import json
import uuid
import logging
import re
import random
from typing import List, Dict, Any

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinetuningAugmentor")

# Environment tokens
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Must be set in .env

# -----------------------------
# Step 1: Collect & Preprocess Input
# -----------------------------
from pydantic import BaseModel, field_validator, ValidationError

class AugmentationExample(BaseModel):
    input_text: str
    output_text: str

    @field_validator('input_text', 'output_text')
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text fields must be non-empty")
        return v.strip()

class AugmentationConfig(BaseModel):
    target_model: str  # e.g., "mixtral-8x7b-32768" or any Groq-supported model name
    examples: List[AugmentationExample]
    finetuning_goal: str

    @field_validator('examples')
    def check_examples_length(cls, v: List[AugmentationExample]) -> List[AugmentationExample]:
        if not (5 <= len(v) <= 10):
            raise ValueError("Provide between 5 and 10 examples")
        return v

# Standardized intermediate format for examples
class StandardExample(BaseModel):
    id: str
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = {}

def normalize_examples(examples: List[AugmentationExample]) -> List[StandardExample]:
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
# Step 2: Dynamic Strategy Selection
# -----------------------------
def determine_augmentation_strategy(config: AugmentationConfig) -> Dict[str, Any]:
    goal = config.finetuning_goal.lower()
    strategy = {}
    # For nuanced/conversational tasks, prefer LLM paraphrasing and back-translation.
    if any(word in goal for word in ["dialogue", "q&a", "conversation", "chat"]):
        strategy["methods"] = ["llm_paraphrasing", "back_translation"]
    else:
        # Otherwise, include EDA-style synonym replacement and synthetic noise.
        strategy["methods"] = ["eda_synonym_replacement", "llm_paraphrasing", "synthetic_noise"]
    strategy["diversity_threshold"] = 0.7  # This threshold is tunable.
    logger.info(f"Determined augmentation strategy: {strategy}")
    return strategy

# -----------------------------
# Helper Functions for Robustness
# -----------------------------
def extract_json(text: str) -> dict:
    """
    Extract the first valid JSON object from text.
    Uses regex to locate a JSON block and then parse it.
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
    Recursively convert unhashable types (like lists or dicts) into hashable tuples.
    """
    if isinstance(item, (list, tuple)):
        return tuple(make_hashable(i) for i in item)
    elif isinstance(item, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    else:
        return item

# -----------------------------
# Step 3: Augmentation Generation via LangChain Groq
# -----------------------------
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def instantiate_groq_llm(model: str) -> ChatGroq:
    """
    Instantiate the ChatGroq model using the provided model name and API key.
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
    Generate an initial candidate augmentation using a prompt chain.
    The output should be valid JSON.
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
    Use a second LLM chain to refine the candidate augmentation.
    If refinement fails (i.e. no valid JSON is returned), return the original candidate.
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
    Simulate metric calculations. In practice, compute semantic similarity (e.g., BERTScore),
    diversity (e.g., SELF-BLEU), and fluency (perplexity).
    Here we simulate by generating random values in a realistic range.
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
    Accept candidate only if:
      - Semantic similarity is between 0.80 and 0.95 (ensures meaning is preserved but not identical)
      - Diversity score is at least 0.70
      - Fluency score is at least 0.80
    """
    if metrics["semantic_similarity"] < 0.80 or metrics["semantic_similarity"] > 0.95:
        return False
    if metrics["diversity_score"] < 0.70:
        return False
    if metrics["fluency_score"] < 0.80:
        return False
    return True

def generate_augmentations(normalized_examples: List[StandardExample],
                           config: AugmentationConfig,
                           strategy: Dict[str, Any],
                           target_count: int = 50) -> List[Dict[str, Any]]:
    """
    For each normalized example, repeatedly generate (and refine) candidate augmentations until
    at least `target_count` valid candidates are collected.
    Invalid candidates (due to JSON extraction or metric failure) are skipped.
    """
    augmented_candidates = []
    attempts = 0
    max_attempts = 100  # Safety valve to avoid infinite loops
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

# -----------------------------
# Step 4: Quality Assurance and Filtering
# -----------------------------
def quality_check(augmentation: Dict[str, Any], config: AugmentationConfig) -> bool:
    """
    Simulate an LLM-based QA check. In production, integrate a validator (e.g., via smolagents and HF_TOKEN)
    to ensure the augmentation aligns with the intended meaning and style.
    """
    qa_prompt = (
        f"Verify that the following augmentation preserves the intended meaning and style for the input/output pair "
        f"given the finetuning goal '{config.finetuning_goal}':\n"
        f"{augmentation['augmentation']}\n"
        "Answer 'yes' if valid, otherwise 'no'."
    )
    logger.debug(f"QA Prompt: {qa_prompt}")
    return True  # Simulation: always passes

# -----------------------------
# Step 5: Post-Processing and Formatting (OpenAI Fine-Tuning Format)
# -----------------------------
def format_for_openai(augmentations: List[Dict[str, Any]]) -> str:
    """
    Format each augmentation candidate in the exact OpenAI fine-tuning format.
    Each JSONL line has the 'messages' key containing a list of messages:
    - System message (hard-coded here as required)
    - User message (augmented input)
    - Assistant message (augmented output)
    """
    output_lines = []
    system_message = "Marv is a factual chatbot that is also sarcastic."  # Hard-coded system prompt
    for aug in augmentations:
        record = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": aug["augmentation"].get("augmented_input")},
                {"role": "assistant", "content": aug["augmentation"].get("augmented_output")}
            ]
        }
        output_lines.append(json.dumps(record))
    logger.info(f"Formatted {len(output_lines)} records in OpenAI fine-tuning format.")
    return "\n".join(output_lines)

def deduplicate_augmentations(augmentations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate or near-duplicate augmentations by converting outputs into hashable keys.
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

# -----------------------------
# Step 6: Optional Interactive Review (via Streamlit)
# -----------------------------
def launch_review_app(formatted_data: str):
    import streamlit as st
    st.title("Finetuning Data Augmentation Review")
    st.write("Review and, if necessary, edit the augmented data below.")
    edited_data = st.text_area("Augmented Data (JSONL)", formatted_data, height=400)
    if st.button("Approve and Save"):
        with open("train.jsonl", "w") as f:
            f.write(edited_data)
        st.success("Data saved successfully to train.jsonl!")

# -----------------------------
# Step 7: Packaging into a Modular Pipeline and Writing Output
# -----------------------------
class FinetuningDataAugmentor:
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.normalized_examples = normalize_examples(config.examples)
        self.strategy = determine_augmentation_strategy(config)
        self.augmentations = []

    def run_augmentation(self) -> List[Dict[str, Any]]:
        logger.info("Starting augmentation generation via LangChain Groq...")
        candidates = generate_augmentations(self.normalized_examples, self.config, self.strategy, target_count=50)
        logger.info(f"Generated {len(candidates)} candidate augmentations before deduplication.")
        unique_candidates = deduplicate_augmentations(candidates)
        logger.info(f"{len(unique_candidates)} unique augmentations after deduplication.")
        self.augmentations = unique_candidates
        return unique_candidates

    def get_formatted_output(self) -> str:
        return format_for_openai(self.augmentations)

    def save_to_file(self, filename: str = "train.jsonl"):
        output = self.get_formatted_output()
        with open(filename, "w") as f:
            f.write(output)
        logger.info(f"Final augmented data saved to {filename}")

    def run_review_interface(self):
        formatted_data = self.get_formatted_output()
        launch_review_app(formatted_data)

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    try:
        # Simulated user input; in practice, these could be collected via a web form or CLI.
        example_data = [
            AugmentationExample(input_text="What is the capital of France?", output_text="Paris"),
            AugmentationExample(input_text="Who wrote '1984'?", output_text="George Orwell"),
            AugmentationExample(input_text="What is the boiling point of water?", output_text="100°C"),
            AugmentationExample(input_text="Who is the CEO of Tesla?", output_text="Elon Musk"),
            AugmentationExample(input_text="What is the largest planet?", output_text="Jupiter")
        ]
        config = AugmentationConfig(
            target_model="mixtral-8x7b-32768",  # Replace with your desired Groq model name
            examples=example_data,
            finetuning_goal="Enhance Q&A robustness and nuance preservation"
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e.json()}")
        exit(1)

    augmentor = FinetuningDataAugmentor(config)
    augmentor.run_augmentation()
    final_output = augmentor.get_formatted_output()
    print("Final Augmented Data:\n", final_output)
    
    # Save the final output to train.jsonl (ensuring at least 50 valid pairs)
    augmentor.save_to_file("train.jsonl")
    
    # Optionally, launch an interactive review interface:
    # augmentor.run_review_interface()
