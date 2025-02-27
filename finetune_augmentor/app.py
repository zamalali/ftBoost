import streamlit as st
from finetune_augmentor.augmentor import AugmentationExample, AugmentationConfig, FinetuningDataAugmentor

# Set page configuration with dark theme
st.set_page_config(
    page_title="Finetuning Data Augmentation Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for dark theme and appealing UI
st.markdown(
    """
    <style>
    /* Dark theme background */
    body {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #2c2c2c;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    h1, h2, h3, h4 {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1.5rem;
        margin-top: 1rem;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        border: 1px solid #555555;
        padding: 0.5rem;
        background-color: #3c3c3c;
        color: #e0e0e0;
    }
    .stNumberInput>div>input {
        border-radius: 5px;
        border: 1px solid #555555;
        padding: 0.5rem;
        background-color: #3c3c3c;
        color: #e0e0e0;
    }
    pre {
        background-color: #333333;
        padding: 1rem;
        border-radius: 5px;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("Finetuning Data Augmentation Generator")

# Step 1: Select Model Type ("openai models" or "gemini models")
model_type = st.selectbox("Select the type of model to finetune", ["openai models", "gemini models"])

# Step 2: Optionally specify a system message (only used for OpenAI)
default_system = "Marv is a factual chatbot that is also sarcastic."
system_message = st.text_input("System Message (optional)", value=default_system)

st.markdown("### Input Schema Template")
if model_type == "openai models":
    st.code(
        '''{
  "messages": [
    {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
  ]
}''', language="json")
else:
    st.code(
        '''{
  "contents": [
    {"role": "user", "parts": [{"text": "What's the capital of France?"}]},
    {"role": "model", "parts": [{"text": "Paris, as if everyone doesn't know that already."}]}
  ]
}''', language="json")

st.markdown("### Enter At Least 3 Input/Output Pairs")
finetuning_goal = st.text_input("Finetuning Goal", "Enhance Q&A robustness and nuance preservation")
num_pairs = st.number_input("Number of Input/Output Pairs", min_value=3, value=3, step=1)

user_pairs = []
for i in range(num_pairs):
    st.markdown(f"#### Pair {i+1}")
    user_msg = st.text_input(f"User Message for Pair {i+1}", key=f"user_{i}")
    assistant_msg = st.text_input(f"Assistant Message for Pair {i+1}", key=f"assistant_{i}")
    user_pairs.append((user_msg, assistant_msg))

# Let the user choose how many augmented pairs to generate (allow as few as 5)
target_augmented = st.number_input("Number of Augmented Pairs to Generate", min_value=5, value=5, step=1)

if st.button("Generate Data"):
    valid_pairs = [(u, a) for u, a in user_pairs if u.strip() and a.strip()]
    if len(valid_pairs) < 3:
        st.error("Please enter at least 3 valid input/output pairs.")
    else:
        # Create AugmentationExample objects from valid pairs
        examples = [AugmentationExample(input_text=u, output_text=a) for u, a in valid_pairs]
        
        # Use a default Groq model name (adjust as necessary)
        target_model = "mixtral-8x7b-32768"
        
        # Create configuration (for Gemini, system_message is ignored)
        try:
            config = AugmentationConfig(
                target_model=target_model,
                examples=examples,
                finetuning_goal=finetuning_goal,
                system_message=system_message
            )
        except Exception as e:
            st.error(f"Configuration error: {e}")
            st.stop()
        
        st.info("Running augmentation pipeline... Please wait.")
        augmentor = FinetuningDataAugmentor(config)
        augmentor.run_augmentation(target_count=target_augmented)
        
        # Format output based on model type
        if model_type.lower() == "openai models":
            output_data = augmentor.get_formatted_output(format_type="openai")
        else:
            output_data = augmentor.get_formatted_output(format_type="gemini")
        
        st.markdown("### Augmented Data")
        st.text_area("Augmented Data", output_data, height=400)
        st.download_button("Download train.jsonl", output_data, file_name="train.jsonl")
        
st.markdown("</div>", unsafe_allow_html=True)
