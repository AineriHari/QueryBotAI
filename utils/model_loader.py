"""
Module for loading and configuring the Google Generative AI model.

This module handles loading environment variables, specifically the Google API key,
and provides functionality to load a specified generative model from Google.

Key Functions:
- load_model: Loads a specified Google Generative AI model using the provided API key.
"""

import os
import logging
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Set up logging
logging.basicConfig(level=logging.INFO)


def load_gemini_model(model_name):
    """
    Loads a specified Google Generative AI model.

    This function retrieves the Google API key from the environment variables, configures
    the Google Generative AI client, and loads the specified model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        genai.GenerativeModel: An instance of the loaded Google Generative AI model.

    Raises:
        ValueError: If the GOOGLE_API_KEY is not found in the environment variables.
    """
    # Load Gemini Model
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    
    # configure Google API Key
    genai.configure(api_key=google_api_key)

    # Load google model
    return genai.GenerativeModel(model_name)


def load_llama_model(model_name):
    """
    Loads a specified Llama 3.2 1B model from Hugging Face locally.

    This function retrieves the Llama model and tokenizer from a local directory
    or Hugging Face if required, and loads the specified model.

    Args:
        model_name (str): The path or Hugging Face model ID to load.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    try:
        # Define quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 8-bit quantization
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto",)
    except Exception as e:
        raise ValueError(f"Error loading model {model_name}: {e}")

    return model, tokenizer
