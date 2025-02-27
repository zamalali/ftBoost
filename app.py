import streamlit as st
from finetune_augmentor import AugmentationExample, AugmentationConfig, FinetuningDataAugmentor
import json
import streamlit.components.v1 as components
from streamlit_ace import st_ace  # Editable code block

# Set page configuration
st.set_page_config(
    page_title="Finetuning Data Augmentation Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a unified dark theme and to style buttons and code blocks
st.markdown(
    """
    <style>
    /* Main content area */
    .block-container {
        background-color: #121212;
        color: #ffffff;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121212;
        color: #ffffff;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Button styling for both Generate Data and Download buttons */
    .stButton>button, .stDownloadButton>button {
        background-color: #808080 !important;
        color: #ffffff !important;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1.5rem;
        margin-top: 1rem;
    }
    /* Text inputs */
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 5px;
        border: 1px solid #ffffff;
        padding: 0.5rem;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stTextArea>textarea {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: "Courier New", monospace;
        border: 1px solid #ffffff;
        border-radius: 5px;
        padding: 1rem;
    }
    /* Header colors */
    h1 {
        color: #00FF00;
    }
    h2, h3, h4 {
        color: #FFFF00;
    }
    /* Field labels */
    label {
        color: #ffffff !important;
    }
    /* Remove extra margin in code blocks */
    pre {
        margin: 0;
    }
    /* Ace editor style overrides */
    .ace_editor {
        border: none !important;
        box-shadow: none !important;
        background-color: #121212 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inject JavaScript to scroll to top on load
components.html(
    """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        setTimeout(function() { window.scrollTo(0, 0); }, 100);
    });
    </script>
    """,
    height=0,
)

# App Title
st.title("Ft Boost Hero 🚀")

st.markdown(
    """
    **Ft Boost Hero** is a powerful tool designed to help you generate high-quality fine-tuning data for AI models. 
    Whether you're working with OpenAI, Gemini, Mistral, or LLaMA models, this app allows you to create structured 
    input-output pairs and apply augmentation techniques to enhance dataset quality. With advanced tuning parameters, 
    semantic similarity controls, and fluency optimization, **Ft Boost Hero** ensures that your fine-tuning data is diverse, 
    well-structured, and ready for training. 🚀
    """,
    unsafe_allow_html=True,
)

# Step 1: Output Format Selection
model_type = st.selectbox(
    "Select the output format for finetuning",
    ["OpenAI Models", "Gemini Models", "Mistral Models", "Llama Models"]
)

# Step 2: System Message (for OpenAI format)
default_system = "Marv is a factual chatbot that is also sarcastic."
system_message = st.text_input("System Message (optional)", value=default_system)

# Display an input schema template based on the selected format
st.markdown("### Input Schema Template")
if model_type == "OpenAI Models":
    st.code(
        '''{
  "messages": [
    {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
  ]
}''', language="json")
elif model_type == "Gemini Models":
    st.code(
        '''{
  "contents": [
    {"role": "user", "parts": [{"text": "What's the capital of France?"}]},
    {"role": "model", "parts": [{"text": "Paris, as if everyone doesn't know that already."}]}
  ]
}''', language="json")
else:
    st.code(
        '''{
  "messages": [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
  ]
}''', language="json")

# Step 3: Input/Output Pairs and Finetuning Goal
st.markdown("##### Enter At Least 3 Input/Output Pairs")
finetuning_goal = "Improve conversational clarity and capture subtle nuances"
st.markdown(f"**Finetuning Goal:** {finetuning_goal}")

# Advanced Tuning Parameters (using fancy sliders)
with st.expander("Show/Hide Advanced Tuning Parameters"):
    min_semantic = st.slider("Minimum Semantic Similarity", 0.0, 1.0, 0.80, 0.01)
    max_semantic = st.slider("Maximum Semantic Similarity", 0.0, 1.0, 0.95, 0.01)
    min_diversity = st.slider("Minimum Diversity Score", 0.0, 1.0, 0.70, 0.01)
    min_fluency = st.slider("Minimum Fluency Score", 0.0, 1.0, 0.80, 0.01)

num_pairs = st.number_input("Number of Pairs", min_value=3, value=3, step=1)

pair_templates = []
for i in range(num_pairs):
    st.markdown(f"#### Pair {i+1}")
    if model_type == "OpenAI Models":
        init_template = ('''{
  "messages": [
    {"role": "system", "content": "''' + (system_message if system_message.strip() else "") + '''"},
    {"role": "user", "content": "Enter your input text here"},
    {"role": "assistant", "content": "Enter your output text here"}
  ]
}''').strip()
        ace_key = f"pair_{i}_{model_type}_{system_message}"
    elif model_type == "Gemini Models":
        init_template = ('''{
  "contents": [
    {"role": "user", "parts": [{"text": "Enter your input text here"}]},
    {"role": "model", "parts": [{"text": "Enter your output text here"}]}
  ]
}''').strip()
        ace_key = f"pair_{i}_{model_type}"
    else:
        init_template = ('''{
  "messages": [
    {"role": "user", "content": "Enter your input text here"},
    {"role": "assistant", "content": "Enter your output text here"}
  ]
}''').strip()
        ace_key = f"pair_{i}_{model_type}"
    
    pair = st_ace(
        placeholder="Edit your code here...",
        value=init_template,
        language="json",
        theme="monokai",
        key=ace_key,
        height=150
    )
    pair_templates.append(pair)

target_augmented = st.number_input("Number of Augmented Pairs to Generate", min_value=5, value=5, step=1)

if st.button("Generate Data"):
    examples = []
    errors = []
    for idx, pair in enumerate(pair_templates):
        try:
            record = json.loads(pair)
            if model_type == "OpenAI Models":
                msgs = record.get("messages", [])
                if len(msgs) != 3:
                    raise ValueError("Expected 3 messages")
                input_text = msgs[1].get("content", "").strip()
                output_text = msgs[2].get("content", "").strip()
            elif model_type == "Gemini Models":
                contents = record.get("contents", [])
                if len(contents) < 2:
                    raise ValueError("Expected at least 2 contents")
                input_text = contents[0]["parts"][0].get("text", "").strip()
                output_text = contents[1]["parts"][0].get("text", "").strip()
            else:
                msgs = record.get("messages", [])
                if len(msgs) != 2:
                    raise ValueError("Expected 2 messages for this format")
                input_text = msgs[0].get("content", "").strip()
                output_text = msgs[1].get("content", "").strip()
            if not input_text or not output_text:
                raise ValueError("Input or output text is empty")
            examples.append(AugmentationExample(input_text=input_text, output_text=output_text))
        except Exception as e:
            errors.append(f"Error in pair {idx+1}: {e}")
    if errors:
        st.error("There were errors in your input pairs:\n" + "\n".join(errors))
    elif len(examples) < 3:
        st.error("Please provide at least 3 valid pairs.")
    else:
        target_model = "mixtral-8x7b-32768"
        try:
            config = AugmentationConfig(
                target_model=target_model,
                examples=examples,
                finetuning_goal=finetuning_goal,
                system_message=system_message,
                min_semantic_similarity=min_semantic,
                max_semantic_similarity=max_semantic,
                min_diversity_score=min_diversity,
                min_fluency_score=min_fluency
            )
        except Exception as e:
            st.error(f"Configuration error: {e}")
            st.stop()
        
        st.markdown('<p style="color: white;">Running augmentation pipeline... Please wait.</p>', unsafe_allow_html=True)
        
        augmentor = FinetuningDataAugmentor(config)
        augmentor.run_augmentation(target_count=target_augmented)
        
        fmt = model_type.lower()
        if fmt == "OpenAI Models":
            output_data = augmentor.get_formatted_output(format_type="openai")
        elif fmt == "Gemini Models":
            output_data = augmentor.get_formatted_output(format_type="gemini")
        elif fmt == "Mistral Models":
            output_data = augmentor.get_formatted_output(format_type="mistral")
        elif fmt == "Llama Models":
            output_data = augmentor.get_formatted_output(format_type="llama")
        else:
            output_data = augmentor.get_formatted_output(format_type="openai")
        
        st.markdown("### Augmented Data")
        st.code(output_data, language="json")
        st.download_button("Download train.jsonl", output_data, file_name="train.jsonl")
