import streamlit as st
from finetune_augmentor import AugmentationExample, AugmentationConfig, FinetuningDataAugmentor
import json
import streamlit.components.v1 as components
from streamlit_ace import st_ace  # Editable code block

# -------------------------------
# Page Configuration and CSS
# -------------------------------
st.set_page_config(
    page_title="Finetuning Data Augmentation Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    /* Button styling */
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
    h1 { color: #00FF00; }
    h2, h3, h4 { color: #FFFF00; }
    /* Field labels */
    label { color: #ffffff !important; }
    /* Remove extra margin in code blocks */
    pre { margin: 0; }
    /* Ace editor style overrides */
    .ace_editor {
        border: none !important;
        box-shadow: none !important;
        background-color: #121212 !important;
    }
    /* Override alert (error/success) text colors */
    [data-testid="stAlert"] { color: #ffffff !important; }
    /* Add white border to expander header */
    [data-testid="stExpander"] > div:first-child {
        border: 1px solid #ffffff !important;
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

# -------------------------------
# App Title and Description
# -------------------------------
st.title("ftBoost 🚀")
st.markdown(
    """
    **ftBoost Hero** is a powerful tool designed to help you generate high-quality fine-tuning data for AI models. 
    Whether you're working with OpenAI, Gemini, Mistral, or LLaMA models, this app allows you to create structured 
    input-output pairs and apply augmentation techniques to enhance dataset quality. With advanced tuning parameters, 
    semantic similarity controls, and fluency optimization, **ftBoost Hero** ensures that your fine-tuning data is diverse, 
    well-structured, and ready for training. 🚀
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Step A: File Upload & Auto-Detection
# -------------------------------
st.markdown("##### Step 1: Upload Your Finetuning data JSONL File if you have one already (Optional)")
uploaded_file = st.file_uploader("Upload your train.jsonl file", type=["jsonl", "txt"])
uploaded_examples = []
detected_model = None

if uploaded_file is not None:
    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        # Auto-detect model type from the first valid snippet
        for line in file_content.splitlines():
            if line.strip():
                record = json.loads(line)
                if "messages" in record:
                    msgs = record["messages"]
                    if len(msgs) >= 3 and msgs[0].get("role") == "system":
                        detected_model = "OpenAI Models"
                    elif len(msgs) == 2:
                        detected_model = "Mistral Models"
                elif "contents" in record:
                    detected_model = "Gemini Models"
                break
        
        # Display an info message based on detection result
        if detected_model is not None:
            st.info(f"This JSONL file format supports the **{detected_model}**.")
        else:
            st.info("The uploaded JSONL file format is not recognized. Please manually select the appropriate model.")
        
        # Process the entire file for valid examples
        for line in file_content.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            input_text, output_text = "", ""
            if "messages" in record:
                msgs = record["messages"]
                if len(msgs) >= 3:
                    input_text = msgs[1].get("content", "").strip()
                    output_text = msgs[2].get("content", "").strip()
                elif len(msgs) == 2:
                    input_text = msgs[0].get("content", "").strip()
                    output_text = msgs[1].get("content", "").strip()
            elif "contents" in record:
                contents = record["contents"]
                if len(contents) >= 2 and "parts" in contents[0] and "parts" in contents[1]:
                    input_text = contents[0]["parts"][0].get("text", "").strip()
                    output_text = contents[1]["parts"][0].get("text", "").strip()
            if input_text and output_text:
                uploaded_examples.append(AugmentationExample(input_text=input_text, output_text=output_text))
        if len(uploaded_examples) < 3:
            st.error("Uploaded file does not contain at least 3 valid input/output pairs.")
        else:
            st.success(f"Uploaded file processed: {len(uploaded_examples)} valid input/output pairs loaded.")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

# -------------------------------
# Step B: Model Selection
# -------------------------------
default_model = detected_model if detected_model is not None else "OpenAI Models"
model_options = ["OpenAI Models", "Gemini Models", "Mistral Models", "Llama Models"]
default_index = model_options.index(default_model) if default_model in model_options else 0
model_type = st.selectbox(
    "Select the output format for finetuning",
    model_options,
    index=default_index
)

# -------------------------------
# Step C: System Message & API Key
# -------------------------------
system_message = st.text_input("System Message (optional) only for OpenAI models", value="Marv is a factual chatbot that is also sarcastic.")
# groq_api_key = st.text_input("LangChain Groq API Key", type="password", help="Enter your LangChain Groq API Key for data augmentation")



groq_api_key = st.text_input(
    "LangChain Groq API Key (if you don't have one, get it from [here](https://console.groq.com/keys))",
    type="password",
    help="Enter your LangChain Groq API Key for data augmentation"
)
# -------------------------------
# Step D: Input Schema Template Display
# -------------------------------
st.markdown("#### Input Schema Template")
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

# -------------------------------
# Step E: Manual Input of Pairs (if no file uploaded)
# -------------------------------
if uploaded_file is None:
    st.markdown("##### Enter at least 3 input/output pairs manually:")
    num_pairs = st.number_input("Number of Pairs", min_value=3, value=3, step=1)
    pair_templates = []
    for i in range(num_pairs):
        st.markdown(f"##### Pair {i+1}")
        if model_type == "OpenAI Models":
            init_template = ('''{
  "messages": [
    {"role": "system", "content": "''' + system_message + '''"},
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

# -------------------------------
# Step F: Augmentation Settings
# -------------------------------
target_augmented = st.number_input("Number of Augmented Pairs to Generate", min_value=5, value=5, step=1)
finetuning_goal = "Improve conversational clarity and capture subtle nuances"
st.markdown(f"**Finetuning Goal:** {finetuning_goal}")

with st.expander("Show/Hide Advanced Tuning Parameters"):
    min_semantic = st.slider("Minimum Semantic Similarity", 0.0, 1.0, 0.80, 0.01)
    max_semantic = st.slider("Maximum Semantic Similarity", 0.0, 1.0, 0.95, 0.01)
    min_diversity = st.slider("Minimum Diversity Score", 0.0, 1.0, 0.70, 0.01)
    min_fluency = st.slider("Minimum Fluency Score", 0.0, 1.0, 0.80, 0.01)

# -------------------------------
# Step G: Generate Data Button and Pipeline Execution
# -------------------------------
if st.button("Generate Data"):
    if not groq_api_key.strip():
        st.error("Please enter your LangChain Groq API Key to proceed.")
        st.stop()
    
    # Choose examples: from uploaded file if available; otherwise from manual input.
    if uploaded_file is not None and len(uploaded_examples) >= 3:
        examples = uploaded_examples
    else:
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
    
    if len(examples) >= 3:
        target_model = "mixtral-8x7b-32768"
        try:
            config = AugmentationConfig(
                target_model=target_model,
                examples=examples,
                finetuning_goal=finetuning_goal,
                groq_api_key=groq_api_key,
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
        if fmt == "openai models":
            output_data = augmentor.get_formatted_output(format_type="openai")
        elif fmt == "gemini models":
            output_data = augmentor.get_formatted_output(format_type="gemini")
        elif fmt == "mistral models":
            output_data = augmentor.get_formatted_output(format_type="mistral")
        elif fmt == "llama models":
            output_data = augmentor.get_formatted_output(format_type="llama")
        else:
            output_data = augmentor.get_formatted_output(format_type="openai")
        
        st.markdown("### Augmented Data")
        st.code(output_data, language="json")
        st.download_button("Download train.jsonl", output_data, file_name="train.jsonl")
