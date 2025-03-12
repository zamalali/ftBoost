# ftBoost - Finetuning Data Augmentation Tool

ftBoost is a robust, scalable pipeline for data augmentation designed for fine-tuning conversational AI models. It reads an input JSONL file (in either OpenAI or Gemini format) and generates augmented data using a locally installed Ollama model via its Python interface. The pipeline produces high-quality augmentations by:

- Running **multiple generation attempts** per training example and selecting the best candidate based on a simulated diversity score.
- Optionally refining the candidate augmentation to further improve semantic accuracy, diversity, and clarity.
- Applying simulated quality metrics and deduplication to filter out poor candidates.
- Outputting the augmented data in the same JSONL format as the input.

The tool supports any model available via Ollama. Simply change the model name using the `--model` parameter. For example, you can use `mistral:latest`, `deepseek-r1:1.5b`, or any other locally pulled model.

---

## Requirements

- **Python 3.7+**
- [Ollama](https://ollama.ai) installed and running locally  
  _Note: Ensure you have pulled your desired model (e.g., `deepseek-r1:1.5b` or `mistral:latest`) via the Ollama CLI._

- Python packages:  
  ```bash
  pip install ollama pydantic
  ```

---

## Input File Format

Your training file (`train.jsonl`) can be in either of the two supported formats:

### OpenAI Format (using "messages" key)
```json
{"messages": [
    {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
    {"role": "user", "content": "what is the principal city of France?"},
    {"role": "assistant", "content": "Paris, undoubtedly"}
]}
```

### Gemini Format (using "contents" key)
```json
{"contents": [
    {"role": "user", "parts": [{"text": "what is the principal city of France?"}]},
    {"role": "model", "parts": [{"text": "Paris, undoubtedly"}]}
]}
```

The pipeline auto-detects the format and outputs augmented data in the same style.

---

## How It Works

### 1. Data Loading and Normalization
The pipeline reads your training file, extracts the user and assistant texts, and normalizes them (e.g., converting to lowercase). Each example is assigned a unique ID.

### 2. Strategy Selection
Based on your fine-tuning goal (passed via `--finetuning_goal`), the pipeline selects a generation strategy. For conversational tasks, it may use methods like `llm_paraphrasing` and `back_translation`; otherwise, it includes additional methods like `eda_synonym_replacement` and `synthetic_noise`.

### 3. Augmentation Generation
For each example, the pipeline:
- Generates an initial augmentation by sending a prompt to the Ollama model.
- Optionally refines the augmentation if you do not skip refinement (controlled by the `--skip_refinement` flag).
- Runs multiple attempts (default is **2**) per example and selects the candidate with the highest simulated diversity score (while ensuring it’s not identical to the original input).

### 4. Quality Metrics and Deduplication
Each candidate augmentation is scored on simulated **semantic similarity**, **diversity**, and **fluency** metrics. Candidates failing the thresholds (configurable via command-line parameters) are rejected. The final candidates are deduplicated based on their augmented input and output.

### 5. Output Formatting
The augmented data is output in **JSONL format**. If your input file uses the "contents" style, the output is formatted in the same style; otherwise, it uses the "messages" style (OpenAI).

### 6. Progress Display
The tool displays an **ASCII banner** and a **progress bar** that updates from 0% to 100% as examples are processed.

---

## Command Line Usage

Below is an example command that runs the pipeline using the **deepseek-r1:1.5b** model (which is known to be fast) and skips the refinement stage for maximum speed:

```bash
python main.py --input data/train.jsonl --target 50 --min_semantic 0.80 --max_semantic 0.95 --min_diversity 0.70 --min_fluency 0.80 --model deepseek-r1:1.5b --finetuning_goal "Improve conversational clarity and capture subtle nuances" --skip_refinement
```

### Parameter Explanation

| Parameter | Description |
|-----------|-------------|
| `--input <PATH>` | Path to your training JSONL file. |
| `--target <NUM>` | Number of augmented pairs to generate. |
| `--min_semantic` & `--max_semantic` | Minimum and maximum thresholds for semantic similarity. |
| `--min_diversity` | Minimum threshold for diversity score. |
| `--min_fluency` | Minimum threshold for fluency score. |
| `--model <MODEL_NAME>` | Specify the Ollama model to use (e.g., `deepseek-r1:1.5b`, `mistral:latest`). |
| `--finetuning_goal "<GOAL>"` | Description of the fine-tuning goal, which guides the augmentation prompts. |
| `--skip_refinement` | If set, the refinement stage is skipped for faster processing. |
| `--output <FILE>` | (Optional) Output file for the augmented data (default: `augmented_train.jsonl`). |

---

## Changing Models

Changing the model is as simple as providing a different model name via the `--model` flag. For example:

### To use Mistral 7B:
```bash
python main.py --input data/train.jsonl --target 50 --model mistral:latest --finetuning_goal "Improve conversational clarity and capture subtle nuances"
```

### To use Deepseek R1 1.5B:
```bash
python main.py --input data/train.jsonl --target 50 --model deepseek-r1:1.5b --finetuning_goal "Improve conversational clarity and capture subtle nuances"
```

_Ensure the specified model has been pulled and is running in your local Ollama instance._

---

## Summary

ftBoost allows you to rapidly generate high-quality augmented data for fine-tuning with a flexible, robust, and highly configurable pipeline. By adjusting thresholds, selecting the appropriate model, and optionally skipping refinement, you can easily balance quality and speed to suit your needs.

**Happy augmenting!** 🎯