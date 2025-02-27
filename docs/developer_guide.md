# Developer Guide

## Introduction
This guide provides developers with insights into the structure, functionality, and extendability of Ft Boost Hero. The application is built using Streamlit for the UI and LangChain Groq for AI-driven augmentation.

## Project Structure
```bash
ftBoost/
├── assets/              # Static assets like icons
├── docs/                # Documentation files
├── app.py               # Main Streamlit application
├── augmentor.py         # Core augmentation logic
├── requirements.txt     # Project dependencies
├── .env                 # API keys (ignored in Git)
└── README.md            # Project overview
```

## Setting Up Development Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/zamalali/ftBoost.git
   cd ftBoost
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Key Modules
### 1. `app.py`
- Handles the frontend via Streamlit.
- Manages user interactions, input handling, and augmentation settings.
- Calls augmentation functions from `augmentor.py`.

### 2. `augmentor.py`
- Implements augmentation logic using LangChain Groq.
- Includes strategies for generating and validating augmented data.
- Supports multiple output formats: OpenAI, Gemini, Mistral, LLama.

## Extending the Application
### Adding New Augmentation Methods
1. Open `augmentor.py`.
2. Define a new function implementing the augmentation strategy.
3. Update `determine_augmentation_strategy()` to include the new method.

### Adding a New Output Format
1. Define a new formatting function in `augmentor.py`.
2. Update the `get_formatted_output()` function to support the new format.
3. Modify `app.py` to include the new format option in the UI.

## Debugging and Logging
- Use `logging` for debugging (`logging.INFO` for general messages, `logging.ERROR` for issues).
- Run Streamlit in debug mode:
  ```bash
  streamlit run app.py --server.runOnSave=true
  ```

## Future Enhancements
- Implement batch processing for large datasets.
- Add a cloud storage option for augmented data.
- Support additional fine-tuning frameworks.

For any issues or contributions, please submit a pull request on GitHub.