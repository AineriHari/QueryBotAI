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
from utils.model_loader import load_model


# Set up logging
logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT = "You are a Python full stack developer with expertise in JavaScript, CSS, HTML, Django, and Python. Respond only with code that includes newly added logic or modified functions, without repeating the entire file structure. When providing code, include only what’s necessary for the update: new functions, modified logic within existing functions, and any additional import statements required to make the additions functional."
USER_PROMPT = "Based on the information from indexing, generate code updates that reflect newly added logic or changes to existing functions. Avoid re-creating the whole file. Focus on additions to the functionality, including only essential new functions and import statements to support these changes. My query is: "


def generate_response(
    query: str,
    documents: list = None,
    model_name: str = "gemini-1.5-flash",
    search_type: str = "code-generation",
):
    """
    Generates a response using the specified generative AI model.

    This function takes a query and a list of documents, loads the specified model, reads
    the content of the documents, and generates a response based on the query.

    Args:
        query (str): The query for which a response is to be generated.
        documents (list, optional): A list of file paths to documents for analysis. Defaults to None.
        model_name (str, optional): The name of the generative AI model to load. Defaults to "gemini-1.5-flash".

    Returns:
        str: The generated response or an error message if no documents are found or if an error occurs.
    """
    logging.info(f"Generating response using model: {model_name}")

    if documents is None or len(documents) == 0:
        raise Exception("No Documents to be loaded for analysis")

    try:
        # store the content with query and related documents data
        chunk_data = ""
        for document in documents:
            if os.path.exists(document):
                with open(document, "rb") as file:
                    file_data = file.read()

                    # Try to decode the file data as UTF-8 (for text-based files)
                    try:
                        decoded_content = file_data.decode("utf-8")
                        chunk_data += decoded_content + "\n"
                    except UnicodeDecodeError:
                        pass

                    if not file_data:
                        logging.warning(f"No content found in {document} for analysis.")

        # prepare the generate response content
        content = (
            [
                f"system role: {SYSTEM_PROMPT}",
                f"User prompt: {USER_PROMPT}{query}",
                f"Information: {chunk_data}",
            ]
            if search_type.lower() == "code-generation"
            else [
                "system role: You are an assistant providing comprehensive answers based on relevant information.",
                f"User prompt: {query}",
                f"Information: {chunk_data}",
            ]
        )

        # Load the Gemini model
        model = load_model(model_name)

        # Generate response using the Gemini model
        response = model.generate_content(content)

        if response and response.text:
            logging.info("Response generated using Gemini model")
            return response.text
        logging.error("The Gemini model did not generate any text response")
        return ""
    except Exception as exc:
        logging.exception(f"Error in Gemini processing: {str(exc)}")
        return ""
