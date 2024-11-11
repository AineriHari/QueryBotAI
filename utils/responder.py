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
import numpy as np
import concurrent.futures
from typing import List, Dict
import google.generativeai as genai
from utils.model_loader import load_model
from utils.preprompts import TextGeneration, CodeGeneration
from termcolor import colored
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer


# Set up logging
logging.basicConfig(level=logging.INFO)


def split_document(content: str, chunk: int = 1024) -> List[str]:
    """
    Splits the given content into smaller chunks of specified size.

    This function divides the input content into multiple chunks, each of a specified length.
    It helps in processing large documents by splitting them into manageable parts.

    Args:
        content (str): The content to be split into chunks.
        chunk (int, optional): The size of each chunk. Defaults to 200.

    Returns:
        List[str]: A list of content chunks.
    """
    return [content[i : i + chunk] for i in range(0, len(content), chunk)]


def read_documents(doc_paths: List[str]) -> Dict:
    """
    Reads and extracts content from a list of document paths.

    This function supports reading text, HTML, Python, CSS, JavaScript, log, PDF, and DOCX files.
    It splits the content into smaller chunks and stores them in a dictionary with the filename as the key.

    Args:
        doc_paths (List[str]): A list of document file paths to read.

    Returns:
        Dict: A dictionary where the keys are filenames and the values are lists of content chunks.
    """
    content = {}
    for path in doc_paths:
        if not os.path.exists(path):
            logging.warning(f"Document not found: {path}")
            continue

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in [".txt", ".html", ".py", ".css", ".js", ".log"]:
                with open(path, "r", encoding="utf-8") as file:
                    content.update(
                        {os.path.basename(path): split_document(file.read())}
                    )
            elif ext == ".pdf":
                with open(path, "rb") as file:
                    reader = PdfReader(file)
                    pdf_content = "\n".join(
                        page.extract_text()
                        for page in reader.pages
                        if page.extract_text()
                    )
                    content.update(
                        {os.path.basename(path): split_document(pdf_content)}
                    )
            elif ext == ".docx":
                doc = Document(path)
                docx_content = "\n".join(para.text for para in doc.paragraphs)
                content.update({os.path.basename(path): split_document(docx_content)})
            else:
                logging.warning(f"Unsupported file type for reading: {path}")
        except Exception as e:
            logging.warning(f"Could not read document {path}: {e}")

    return content


def get_relevant_chunk_for_query(
    embedding_model: SentenceTransformer,
    query_embedding: np.ndarray,
    document_chunks: List[str],
    top_n: int = 3,
) -> List[str]:
    """
    Retrieves the top relevant chunks from the document based on the precomputed query embedding.

    Args:
        embedding_model (SentenceTransformer): The embedding model used for encoding.
        query_embedding (np.ndarray): The precomputed embedding of the query.
        document_chunks (List[str]): A list of document chunks to search from.
        top_n (int, optional): The number of top relevant chunks to return. Defaults to 3.

    Returns:
        List[str]: A list of the most relevant document chunks.
    """
    # Encode the document chunks
    chunk_embeddings = embedding_model.encode(document_chunks).astype(np.float32)

    # Compute cosine similarity between the query and all document chunks
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get indices of the top `n` relevant chunks
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # Retrieve the top `n` relevant chunks based on indices
    return [document_chunks[idx] for idx in top_indices]


def process_document_chunks(
    embedding_model: SentenceTransformer,
    query_embedding: np.ndarray,
    chunks: List[str],
    top_n: int = 3
) -> List[str]:
    """
    Process chunks of a single document in parallel to get relevant chunks.

    Args:
        embedding_model (SentenceTransformer): The embedding model used for encoding.
        query_embedding (np.ndarray): The precomputed embedding of the query.
        chunks (List[str]): A list of chunks within a single document.
        top_n (int): The number of top relevant chunks to retrieve.

    Returns:
        List[str]: A list of relevant chunks for the document.
    """
    # Split the chunks into smaller groups for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(get_relevant_chunk_for_query, embedding_model, query_embedding, [chunk], top_n)
            for chunk in chunks
        ]
        # Collect results
        relevant_chunks = []
        for future in futures:
            relevant_chunks.extend(future.result())
    return relevant_chunks


def process_documents(
    embedding_model: SentenceTransformer,
    query: str,
    document_content: Dict[str, List[str]],
    top_n: int = 3
) -> Dict[str, List[str]]:
    """
    Process multiple documents in parallel to get relevant chunks.

    Args:
        embedding_model (SentenceTransformer): The embedding model used for encoding.
        query (str): The user query.
        document_content (Dict[str, List[str]]): Dictionary where keys are filenames and values are lists of chunks.
        top_n (int): The number of top relevant chunks to retrieve.

    Returns:
        Dict[str, List[str]]: Dictionary of relevant chunks for each document.
    """
    query_embedding = embedding_model.encode(query).astype(np.float32)
    relevant_contents = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            filename: executor.submit(process_document_chunks, embedding_model, query_embedding, chunks, top_n)
            for filename, chunks in document_content.items()
        }
        for filename, future in futures.items():
            relevant_contents[filename] = future.result()
    
    print(f"relevant contents: {relevant_contents}")
    return relevant_contents


def generate_response(
    embedding_model: SentenceTransformer,
    query: str,
    documents: List[str] = None,
    model_name: str = "gemini-1.5-flash",
    search_type: str = "code-generation",
):
    """
    Generates a response using the specified generative AI model based on the user query and documents.

    This function reads the content of the documents, extracts relevant chunks, and sends
    the query along with the relevant information to the generative AI model to generate a response.

    Args:
        embedding_model (SentenceTransformer): The embedding model for encoding the query and documents.
        query (str): The user query for which a response is to be generated.
        documents (List[str], optional): A list of document paths to be analyzed. Defaults to None.
        model_name (str, optional): The name of the generative AI model to use. Defaults to "gemini-1.5-flash".
        search_type (str, optional): The type of response generation (e.g., "code-generation"). Defaults to "code-generation".

    Returns:
        str: The generated response text or an error message if the response could not be generated.
    """
    logging.info(f"Generating response using model: {model_name}")

    if documents is None or len(documents) == 0:
        logging.error("No Documents to be loaded for analysis")
        return ""

    # Read and prepare content from documents
    document_content = read_documents(documents)

    if not all(list(document_content.values())):
        logging.warning("No readable content found in provided documents.")
        return ""

    # Load the relavent documents and extract the relavant content
    relevant_contents = process_documents(embedding_model=embedding_model, query=query, document_content=document_content)

    # Select appropriate prompt template
    if search_type.lower() == "code-generation":
        prompt_template = CodeGeneration()
        user_prompt = prompt_template.USER_PROMPT.format(query=query)
    else:
        prompt_template = TextGeneration()
        user_prompt = prompt_template.USER_PROMPT.format(query=query)

    # Prepare content for model input
    parse_relevant_information = "\n".join(
        [
            f"Document: {filename}\ncontent:{' '.join(chunk for chunk in relevant_contents[filename])}"
            for filename in relevant_contents
        ]
    )
    content = [
        f"System Prompt: {prompt_template.SYSTEM_PROMPT}",
        f"User Prompt: {user_prompt}",
        f"Relevant Information for User Prompt: {parse_relevant_information}",
    ]

    # Load the Gemini model
    try:
        # create a generative config
        generation_config = genai.types.GenerationConfig(
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
            try:
                print(colored(chunk.text, "green"))
                final_response += chunk.text
            except Exception as _:
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
