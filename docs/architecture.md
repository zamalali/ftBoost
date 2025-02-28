# Architecture

## Overview
ftBoost is built using Streamlit and LangChain Groq to facilitate AI-driven data augmentation for fine-tuning conversational models. The architecture is designed for scalability, modularity, and ease of use.

## Components

### 1. **Frontend (Streamlit UI)**
- Provides an intuitive interface for users to configure augmentation settings.
- Allows users to input data, select output formats, and adjust augmentation parameters.
- Displays real-time augmentation results and allows JSONL downloads.

### 2. **Backend (Augmentation Pipeline)**
- Processes user input and normalizes it for augmentation.
- Uses LangChain Groq to generate, refine, and validate augmented data.
- Supports multiple output formats (OpenAI, Gemini, Mistral, LLama).

### 3. **Configuration & Storage**
- Uses environment variables (`.env`) for API keys.
- Stores user-selected parameters for augmentation.
- Outputs generated JSONL files for fine-tuning purposes.

## Data Flow
1. **User Input:** Users enter input-output pairs and configure augmentation settings.
2. **Processing:** The augmentation pipeline applies predefined strategies to generate new data.
3. **Refinement & Validation:** Augmented data is validated based on semantic similarity, diversity, and fluency.
4. **Output:** The processed data is formatted into JSONL and provided for download.

## Technologies Used
- **Streamlit**: Interactive web-based UI.
- **LangChain Groq**: AI-driven text augmentation.
- **Python**: Core programming language.
- **Pydantic**: Data validation and normalization.

## Future Enhancements
- Support for additional model formats.
- Improved augmentation strategies using fine-tuned LLMs.
- Cloud-based storage and retrieval of augmented datasets.