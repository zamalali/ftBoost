import gradio as gr
from finetune_augmentor import AugmentationExample, AugmentationConfig, FinetuningDataAugmentor
import json

# Define the Gradio app
def generate_data(model_type, system_message, num_pairs, pairs, target_augmented, min_semantic, max_semantic, min_diversity, min_fluency):
    examples = []
    errors = []
    
    for pair in pairs:
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
                    raise ValueError("Expected 2 messages")
                input_text = msgs[0].get("content", "").strip()
                output_text = msgs[1].get("content", "").strip()

            if not input_text or not output_text:
                raise ValueError("Input or output text is empty")

            examples.append(AugmentationExample(input_text=input_text, output_text=output_text))
        except Exception as e:
            errors.append(f"Error in pair: {e}")
    
    if errors:
        return f"Errors: {errors}", None
    elif len(examples) < 3:
        return "Please provide at least 3 valid pairs.", None
    
    target_model = "mixtral-8x7b-32768"
    try:
        config = AugmentationConfig(
            target_model=target_model,
            examples=examples,
            finetuning_goal="Improve conversational clarity and capture subtle nuances",
            system_message=system_message,
            min_semantic_similarity=min_semantic,
            max_semantic_similarity=max_semantic,
            min_diversity_score=min_diversity,
            min_fluency_score=min_fluency
        )
    except Exception as e:
        return f"Configuration error: {e}", None
    
    augmentor = FinetuningDataAugmentor(config)
    augmentor.run_augmentation(target_count=target_augmented)
    
    output_data = augmentor.get_formatted_output(format_type=model_type.lower())
    return "Augmentation complete!", output_data

with gr.Blocks(theme="JohnSmith9982/small_and_pretty") as demo:
    gr.Markdown("""# Ft Boost Hero 🚀
    Generate high-quality fine-tuning data for AI models like OpenAI, Gemini, Mistral, and LLaMA. 
    Apply augmentation techniques to enhance dataset quality.
    """)
    
    with gr.Row():
        model_type = gr.Dropdown(["OpenAI Models", "Gemini Models", "Mistral Models", "Llama Models"], label="Select Output Format")
        system_message = gr.Textbox(label="System Message (optional)", value="Marv is a factual chatbot that is also sarcastic.")
        num_pairs = gr.Number(label="Number of Pairs", value=3, minimum=3)
    
    gr.Markdown("### Input Schema Template")
    schema_output = gr.Code("""
    {
      "messages": [
        {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
      ]
    }
    """, language="json")
    
    pairs = gr.Textbox(label="Enter JSON Input/Output Pairs", placeholder="Paste JSON pairs here", interactive=True, lines=10)
    target_augmented = gr.Number(label="Number of Augmented Pairs", value=5, minimum=5)
    
    with gr.Accordion("Advanced Tuning Parameters"):
        min_semantic = gr.Slider(label="Min Semantic Similarity", minimum=0.0, maximum=1.0, value=0.8)
        max_semantic = gr.Slider(label="Max Semantic Similarity", minimum=0.0, maximum=1.0, value=0.95)
        min_diversity = gr.Slider(label="Min Diversity Score", minimum=0.0, maximum=1.0, value=0.7)
        min_fluency = gr.Slider(label="Min Fluency Score", minimum=0.0, maximum=1.0, value=0.8)
    
    with gr.Row():
        generate_button = gr.Button("Generate Data")
        download_button = gr.File(label="Download train.jsonl")
    
    status_output = gr.Textbox(label="Status")
    output_json = gr.JSON(label="Augmented Data")
    
    generate_button.click(generate_data, 
                         inputs=[model_type, system_message, num_pairs, pairs, target_augmented, min_semantic, max_semantic, min_diversity, min_fluency], 
                         outputs=[status_output, output_json])
    
    gr.HTML('<div style="text-align: center; margin-top: 20px;">Made with ❤️</div>')

if __name__ == "__main__":
    demo.queue().launch()
