import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM


# Load environment variables from .env file
load_dotenv(override=True)

# Model ID for Llama 3.2 1B
model_id = "meta-llama/Llama-3.2-1B"
local_directory = './llama3.2-1b-model'

# Create the local directory if it doesn't exist
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

# Download model and tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=local_directory,
    token=os.getenv("HUGGINGFACE_API_TOKEN")
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=local_directory,
    low_cpu_mem_usage=True
)