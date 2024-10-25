"""
Module for loading and configuring the Google Generative AI model.

This module handles loading environment variables, specifically the Google API key,
and provides functionality to load a specified generative model from Google.

Key Functions:
- load_model: Loads a specified Google Generative AI model using the provided API key.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment variables from .env file
load_dotenv()


def load_model(model_name):
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
