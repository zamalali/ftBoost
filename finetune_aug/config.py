# Loads .env variables and configuration settings
# finetune_aug/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

LANGCHAIN_GROQ_API_KEY = os.getenv("LANGCHAIN_GROQ_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
AUGMENTATION_TARGET_COUNT = int(os.getenv("AUGMENTATION_TARGET_COUNT", "80"))
