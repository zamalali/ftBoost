![Logo](assets/icon.png)

# Ft Boost Hero 🚀

Ft Boost Hero is a robust, interactive data augmentation generator built with Streamlit. It is designed to create augmented data for fine-tuning conversational models in various JSONL formats (OpenAI, Gemini, Mistral, and LLama). The application uses LangChain Groq for LLM-based data generation and offers advanced tuning options for semantic similarity, diversity, and fluency.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Code Structure](#architecture--code-structure)
  - [app.py](#apppy)
  - [augmentor.py](#augmentorpy)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)
- [License](#license)

---

## Overview

Ft Boost Hero allows users to quickly generate augmented conversational data for fine-tuning various language models. The app supports multiple output formats:
- **OpenAI models:** Uses a three-message format including a system message.
- **Gemini models:** Uses a contents array with distinct user and model parts.
- **Mistral & LLama models:** Uses a simplified two-message format (user and assistant) with dynamic content flattening to remove unwanted keys.

Users can also tune advanced parameters such as semantic similarity, diversity, and fluency scores via interactive sliders.

---

## Features

- **Multi-format Output:** Choose between OpenAI, Gemini, Mistral, and LLama JSONL formats.
- **Dynamic Code Blocks:** Editable Ace editor blocks for each input/output pair.
- **Advanced Tuning Parameters:** Fine-tune augmentation parameters using sliders.
- **LLM Augmentation Pipeline:** Leverages LangChain Groq for generating and refining augmentations.
- **Dynamic Strategy Selection:** Automatically selects augmentation methods based on the fine-tuning goal.
- **Interactive Review & Download:** Preview augmented JSONL output and download it as a file.

---

## Architecture & Code Structure

### app.py

The `app.py` file is the main Streamlit application. It handles:
- **User Interface:**  
  - Logo display, page configuration, and custom CSS for a dark theme.
  - A dropdown to select the desired output format (OpenAI, Gemini, Mistral, LLama).
  - Input fields for system messages (used in OpenAI format) and sample JSON schema display.
  - Editable code blocks (via the Ace editor) for entering input/output pairs.
  - An "Advanced Tuning Parameters" section with sliders for semantic similarity, diversity, and fluency.
- **Data Processing:**  
  - Parsing user-entered JSON and validating the input pairs.
  - Passing user configuration (including advanced parameters) to the augmentation pipeline.
- **Augmentation & Output:**  
  - Initiates the augmentation pipeline from `augmentor.py` with the configured settings.
  - Dynamically formats the output according to the selected output format.
  - Provides a preview of the augmented data and a download button for the JSONL file.

### augmentor.py

The `augmentor.py` file contains the core logic for generating augmented data. Key components include:

- **Data Models & Preprocessing:**  
  - `AugmentationExample` and `AugmentationConfig` are defined using Pydantic to validate inputs.
  - `normalize_examples` converts raw examples to a standardized format.
- **Dynamic Strategy Selection:**  
  - The augmentation strategy (e.g., methods such as `llm_paraphrasing` and `back_translation`) is chosen based on the fine-tuning goal.
- **Helper Functions:**  
  - JSON extraction and validation utilities.
  - **Content Formatting Helpers:**  
    - `fix_content` converts Python dictionary-like strings (with single quotes) into valid JSON.
    - `flatten_content` flattens JSON strings representing dictionaries into a plain-text string by joining their values.
- **Augmentation Generation Pipeline:**  
  - Utilizes LangChain Groq (`ChatGroq` and `ChatPromptTemplate`) to generate initial candidate augmentations and then refines them.
  - Simulated metric calculations ensure quality by checking semantic similarity, diversity, and fluency against user-defined thresholds.
- **Output Formatting:**  
  - The pipeline formats the final output in one of four supported JSONL structures:
    - **OpenAI:** Includes system, user, and assistant messages.
    - **Gemini:** Uses a contents array with separate parts.
    - **Common (Mistral/LLama):** Uses a simplified two-message format.
- **Pipeline Class:**  
  - The `FinetuningDataAugmentor` class encapsulates the full augmentation process, including deduplication and saving of results.

---

## Installation

### Clone the Repository:

```bash
git clone https://github.com/yourusername/ft-boost-hero.git
cd ft-boost-hero
```

### Create and Activate a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Configure Environment Variables:

Create a `.env` file in the project root with your API keys:

```env
HF_TOKEN=your_hf_token_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## Usage

### Run the Streamlit App:

```bash
streamlit run app.py
```

### Interact with the App:

1. Select the desired output format (OpenAI, Gemini, Mistral, or LLama).
2. (For OpenAI) Enter the system message.
3. View the sample input schema.
4. Enter at least three input/output pairs using the editable Ace editor blocks.
5. Optionally, expand the "Advanced Tuning Parameters" section and adjust the sliders.
6. Click **Generate Data** to run the augmentation pipeline.
7. Preview the generated augmented JSONL output and download it.

---

## Configuration

### Target Model:
The augmentation pipeline always uses `mixtral-8x7b-32768` as the target model for generation.

### Tuning Parameters:
The following parameters can be adjusted via sliders:

- **Minimum Semantic Similarity**
- **Maximum Semantic Similarity**
- **Minimum Diversity Score**
- **Minimum Fluency Score**

### Output Formats:
The app supports dynamic output formatting:

- **OpenAI Models:** Three-message format (system, user, assistant).
- **Gemini Models:** Contents array with user and model parts.
- **Mistral & LLama:** Common two-message format with content flattening.

---

## Directory Structure

```bash
ft-boost-hero/
├── assets/
│   └── icon.png           # Repository logo
├── app.py                 # Main Streamlit application
├── finetune_augmentor.py  # Augmentation pipeline (augmentor.py)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
└── README.md              # This README file
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
