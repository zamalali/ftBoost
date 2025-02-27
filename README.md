<!-- ![Logo](assets/icon.png) -->
<img src="assets/readme.png" alt="Logo" width="700">

# Ft Boost Hero 🚀

Ft Boost Hero is a powerful data augmentation tool designed to streamline fine-tuning for conversational AI models. Built with Streamlit, it enables users to generate high-quality augmented data in multiple JSONL formats (OpenAI, Gemini, Mistral, and LLama). The tool leverages LangChain Groq for AI-powered augmentation and allows customization through advanced tuning options.

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

Ft Boost Hero helps users efficiently generate augmented conversational data for fine-tuning language models. It supports multiple formats:
- **OpenAI:** Uses a structured three-message format.
- **Gemini:** Organizes content into a structured array.
- **Mistral & LLama:** Adopts a simplified two-message format.

The app includes an intuitive UI with interactive sliders to adjust semantic similarity, diversity, and fluency.

---

## Features

- **Multi-format support:** Generate JSONL data for OpenAI, Gemini, Mistral, and LLama models.
- **Customizable augmentation:** Adjust parameters for enhanced data quality.
- **Efficient AI-driven augmentation:** Uses LangChain Groq to create and refine augmented samples.
- **Intuitive user experience:** Preview and download generated datasets easily.

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
git clone https://github.com/yourusername/ft-boost-hero.git
cd ft-boost-hero
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
