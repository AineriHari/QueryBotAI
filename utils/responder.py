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
from typing import List
import google.generativeai as genai
from utils.model_loader import load_model
from utils.preprompts import TextGeneration, CodeGeneration
from termcolor import colored


# Set up logging
logging.basicConfig(level=logging.INFO)


def generate_response(
    query: str,
    documents: List[str] = None,
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
        raise ValueError("No Documents to be loaded for analysis")

    def read_documents(doc_paths: List[str]) -> str:
        content = ""
        for path in doc_paths:
            if not os.path.exists(path):
                logging.warning(f"Document not found: {path}")
                continue
            with open(path, "rb") as file:
                file_data = file.read()
                try:
                    decoded_content = file_data.decode("utf-8")
                    content += (
                        f"\n[Document: {os.path.basename(path)}]\n{decoded_content}\n"
                    )
                except UnicodeDecodeError:
                    logging.warning(f"Skipping non-text content in {path}")
        return content

    # Read and prepare content from documents
    document_content = read_documents(documents)

    if not document_content:
        logging.warning("No readable content found in provided documents.")

    # Select appropriate prompt template
    if search_type.lower() == "code-generation":
        prompt_template = CodeGeneration()
        user_prompt = (
            prompt_template.USER_PROMPT.format(query=query)
            + "\nExisting Code:\n"
            + document_content.strip()
        )
    else:
        prompt_template = TextGeneration()
        user_prompt = prompt_template.USER_PROMPT.format(query=query)

    # Prepare content for model input
    content = [
        f"System Prompt: {prompt_template.SYSTEM_PROMPT}",
        f"User Prompt: {user_prompt}",
        f"Information: {document_content.strip()}",
    ]

    # Load the Gemini model
    try:
        # create a generative config
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        model = load_model(model_name)

        # get the response
        if search_type == "code-generation":
            response = model.generate_content(
                content,
                generation_config=generation_config,
                stream=True,
                tools="code_execution",
            )
        else:
            response = model.generate_content(
                content, generation_config=generation_config, stream=True
            )

        final_response = ""
        for chunk in response:
            if chunk is not None and hasattr(chunk, "text") and chunk.text:
                print(colored(chunk.text, "green"))
                final_response += chunk.text
            else:
                # Log warning if the chunk is empty and log finish_reason if available
                finish_reason = getattr(chunk, "finish_reason", None)
                logging.warning(
                    f"Received an empty chunk. Finish reason: {finish_reason} - Skipping..."
                )

        if final_response:
            logging.info("Response generated successfully")
            return final_response

        logging.error("The Gemini model did not generate any text response")
        return ""

    except Exception as exc:
        logging.exception(f"Error in model processing: {exc}")
        return ""
