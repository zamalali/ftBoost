<!-- ![Logo](assets/icon.png) -->
<p align="center">
  <img src="assets/icon.png" alt="Logo" width="500">
</p>

# ftBoost 🐦‍⬛

ftBoost is a powerful data augmentation tool designed to streamline fine-tuning for conversational AI models. Built with Streamlit, it enables users to generate high-quality augmented data in multiple JSONL formats (OpenAI, Gemini, Mistral, and LLama). The tool leverages LangChain Groq for AI-powered augmentation and allows customization through advanced tuning options.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

ftBoost helps users efficiently generate augmented conversational data for fine-tuning language models. It supports multiple formats:
- **OpenAI:** Uses a structured three-message format.
- **Gemini:** Organizes content into a structured array.
- **Mistral & LLama:** Adopts a simplified two-message format.

The app includes an intuitive UI with interactive sliders to adjust semantic similarity, diversity, and fluency.

---

## Features

**Multi-format Output:**  
- Generates augmented data in OpenAI, Gemini, Mistral, and LLama JSONL formats.

**Robust Data Modeling and Validation:**  
- Uses Pydantic models to validate input/output pairs ensuring non-empty texts.
- Enforces a minimum of 3 examples per augmentation run.

**Normalization & Standardization:**  
- Normalizes examples by lowercasing and adding unique IDs.
- Captures metadata such as the original word count for further processing.

**Dynamic Strategy Selection:**  
- Inspects the finetuning goal to choose between methods like LLM paraphrasing, back translation, synonym replacement, and synthetic noise.
- Adjusts strategy based on whether the goal is conversational or general.

**AI-Driven Augmentation Pipeline:**  
- **Initial Augmentation:**  
  - Generates candidate augmentations using a LangChain Groq-powered LLM prompt.
  - Ensures the output is in a valid JSON format with keys `augmented_input` and `augmented_output`.
- **Refinement Process:**  
  - Refines the candidate using a second LLM prompt chain to maximize semantic accuracy, diversity, and clarity.
  - Falls back to the original candidate if refinement fails.

**Quality and Metric Validation:**  
- Simulates metrics for semantic similarity, diversity, and fluency.
- Accepts only candidates that meet predefined thresholds.
- Performs a simulated quality check to verify that the augmentation preserves the intended meaning and style.

**Deduplication:**  
- Removes duplicate augmentations by hashing normalized input and output pairs, ensuring a unique dataset.

**Flexible Output Formatting:**  
- Formats data for OpenAI (three-message structure), Gemini (content array), and common formats for Mistral/LLama.
- Validates JSONL records before final output.

**Streamlit-based Interactive Interface:**  
- Supports file uploads with auto-detection of format (OpenAI, Gemini, etc.).
- Provides an Ace editor for manual entry if no file is uploaded.
- Offers real-time parameter adjustments via interactive sliders.
- Includes a download button to save the final augmented dataset.

---

## Usage

1. **Select output format** (OpenAI, Gemini, Mistral, or LLama).
2. **Enter system message** (for OpenAI format).
3. **Provide input/output examples** using interactive fields.
4. **Adjust augmentation parameters** (semantic similarity, diversity, fluency) as needed.
5. **Generate augmented data** using the AI pipeline.
6. **Review and download** the final dataset.

---

## Installation

### Clone the Repository:
```bash
git clone https://github.com/zamalali/ftBoost
cd ftBoost-hero
```

### Set Up Environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure API Keys:
Create a `.env` file and add your credentials:
```env
HF_TOKEN=your_hf_token_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## Configuration

- **Target Model:** `mixtral-8x7b-32768` (default model for augmentation).
- **Customizable Tuning Parameters:**
  - Minimum/Maximum Semantic Similarity
  - Minimum Diversity Score
  - Minimum Fluency Score
- **Output Formats:**
  - OpenAI: Structured three-message format.
  - Gemini: Content array structure.
  - Mistral & LLama: Simplified two-message format.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
