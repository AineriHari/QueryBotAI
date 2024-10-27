"""
Module for downloading and saving a pretrained model from Hugging Face's Transformers library.

This module specifically handles the downloading of the 'sentence-transformers/paraphrase-MiniLM-L6-v2' model,
including its configuration, model weights, and tokenizer files. It saves these files to a specified local directory
and optionally copies them to a Hugging Face cache directory.

Key Operations:
- Download model configuration.
- Download model weights.
- Download tokenizer files.
- Save all files to a local directory and copy to a cache location.
"""
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import shutil

# Define the model name and the local directory where we want to store the files
trasnformers_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
local_directory = './paraphrase-MiniLM-L6-v2-local'

# Create the local directory if it doesn't exist
if not os.path.exists(local_directory):
    os.makedirs(local_directory)


def download_and_save_model(model_name, local_dir):
    """
    Downloads and saves the specified pretrained model's configuration, weights, and tokenizer files.

    Args:
        model_name (str): The name of the model to download from Hugging Face.
        local_dir (str): The local directory to save the downloaded model files.
    """
    # Download model config and save it locally
    logging.info("Downloading config.json...")
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(local_dir)

    # Download model weights and save them locally
    logging.info("Downloading model weights...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_dir)

    # Download tokenizer and save it locally
    logging.info("Downloading tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_dir)

    logging.info(f"Model files downloaded to: {local_dir}")


def copy_to_cache(local_dir, cache_dir):
    """
    Copies the downloaded model files from the local directory to the Hugging Face cache directory.

    Args:
        local_dir (str): The local directory where the model files are saved.
        cache_dir (str): The directory where the model files will be copied.
    """
    # Create the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Copy all downloaded files to the cache directory
    for filename in os.listdir(local_dir):
        shutil.copy(os.path.join(local_dir, filename), os.path.join(cache_dir, filename))

    logging.info(f"Model files copied to cache: {cache_dir}")


try:
    # Execute the downloading and saving of the model
    download_and_save_model(trasnformers_model_name, local_directory)

    # Define cache directory
    cache_directory = os.path.expanduser(
        '~/.cache/huggingface/transformers/'
        'sentence-transformers__paraphrase-MiniLM-L6-v2'
    )

    # Copy files to the cache directory
    copy_to_cache(local_directory, cache_directory)
except Exception as exc:
    logging.exception(str(exc))
    logging.error("Failed to setup the local Transformation setup.")
