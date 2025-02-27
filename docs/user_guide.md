# User Guide

## Introduction
Ft Boost Hero is a user-friendly tool for generating augmented datasets for fine-tuning AI models. This guide will help you navigate its features and optimize your workflow.

## Getting Started
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zamalali/ftBoost.git
   cd ftBoost
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up API keys in `.env`:
   ```env
   HF_TOKEN=your_hf_token_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

## Using the Interface
### 1. Selecting an Output Format
- Choose between OpenAI, Gemini, Mistral, or LLama models.
- This determines the structure of the generated JSONL data.

### 2. Configuring Input Data
- Provide at least **three input/output pairs** for augmentation.
- Use the interactive editor to enter data in JSON format.

### 3. Adjusting Augmentation Parameters
- Fine-tune semantic similarity, diversity, and fluency scores using sliders.
- These parameters control the quality and variability of augmented data.

### 4. Generating and Downloading Data
- Click **Generate Data** to run the augmentation pipeline.
- Review the generated samples in the preview section.
- Download the JSONL file for fine-tuning your AI model.

## Troubleshooting
- If you encounter API errors, check your `.env` file for correct credentials.
- If augmentation results are unsatisfactory, try adjusting tuning parameters.
- Restart the app if UI elements do not update correctly.

## FAQs
### **How many input/output pairs are needed?**
At least **three** pairs are required for effective augmentation.

### **Can I use this tool for non-conversational datasets?**
Yes, but it is optimized for dialogue-based fine-tuning.

### **How do I reset all inputs?**
Refresh the page or restart the app.

## Support
For bug reports or feature requests, open an issue on [GitHub](https://github.com/zamalali/ftBoost/issues).
