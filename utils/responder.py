"""
Module for generating responses using the Gemini generative AI model.

This module provides functionality to generate responses based on a user query and a set
of documents. It loads the specified generative AI model, processes the documents, and
generates a response using the provided query.

Key Functions:
- generate_response: Generates a response using the Gemini model based on the provided query and documents.
"""

import os
import logging
from utils.model_loader import load_gemini_model, load_llama_model


# Set up logging
logging.basicConfig(level=logging.INFO)


def generate_response(
        query: str,
        documents: list = None,
        model_name: str = "gemini-1.5-flash",
        model_type: str = "gemini"
):
    """
    Generates a response using the specified generative AI model.

    This function takes a query and a list of documents, loads the specified model, reads
    the content of the documents, and generates a response based on the query.

    Args:
        query (str): The query for which a response is to be generated.
        documents (list, optional): A list of file paths to documents for analysis. Defaults to None.
        model_name (str, optional): The name of the generative AI model to load. Defaults to "gemini-1.5-flash".
        model_type (str, optional): The type of generative AI model to use, either "gemini" or "llama".

    Returns:
        str: The generated response or an error message if no documents are found or if an error occurs.
    """
    logging.info(f"Generating response using model: {model_name}")

    if documents is None or len(documents) == 0:
        return "No Documents to be loaded for analysis"

    try:
        # store the content with query and related documents data
        content = [
            "system role: You are an assistant providing comprehensive answers based on relevant information.",
            query,
        ]
        for document in documents:
            if os.path.exists(document):
                with open(document, 'rb') as file:
                    file_data = file.read()

                    # Try to decode the file data as UTF-8 (for text-based files)
                    try:
                        decoded_content = file_data.decode('utf-8')
                        content.append(decoded_content)
                    except UnicodeDecodeError:
                        pass

                    if not file_data:
                        logging.warning(f"No content found in {document} for analysis.")

        if model_type == "gemini":
            # Load the Gemini model
            model = load_gemini_model(model_name)

            # Generate response using the Gemini model
            response = model.generate_content(content)

            if response and response.text:
                logging.info("Response generated using Gemini model")
                return response.text
            else:
                return "The Gemini model did not generate any text response"
        else:
            # Load the Llama-3.2-1B model
            model, tokenizer = load_llama_model(model_name)

            # Prepare the input text for Llama 2
            input_text = "\n\n".join(content)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True,  max_length=512)

            # Generate response using Llama-3.2-1B model
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if response_text:
                logging.info("Response generated using Llama-3.2-1B model")
                return response_text
            else:
                return "The Llama-3.2-1B model did not generate any text response"
    except Exception as exc:
        logging.exception(f"Error in processing: {str(exc)}")
        return f"An error occurred while processing the document: {str(exc)}"
