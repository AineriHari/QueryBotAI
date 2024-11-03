"""
Module for retrieving documents based on user queries using FAISS.

This module provides functionality to retrieve relevant documents from a FAISS index
using embeddings generated from user queries. It handles document management by copying
retrieved documents to a designated folder for easy access.

Key Functions:
- retrieve_documents: Retrieves relevant documents for a given query using a FAISS index.
"""

import os
import time
import numpy as np
import traceback
from typing import Tuple
import google.generativeai as genai
from utils.model_loader import load_model
from faiss.swigfaiss import IndexFlatL2
import logging
import json
from sentence_transformers import SentenceTransformer


# Set up logging
logging.basicConfig(level=logging.INFO)


def analyze_chunk_with_llm(
    model: genai.GenerativeModel, chunk: bytes, query: str, max_retries: int = 2
) -> Tuple[bool, bytes]:
    """
    Analyzes a text chunk to determine its relevance to a user query using a Generative AI model.

    Args:
        model (genai.GenerativeModel): An instance of the GenerativeModel used to generate responses.
        chunk (bytes): The text chunk that needs to be analyzed for relevance.
        query (str): The user question to which the relevance of the text chunk will be evaluated.
        max_retries (int): Number of retries allowed if relevance is unclear.

    Returns:
        Tuple[bool, bytes]: A tuple where the first element is a boolean indicating
                            whether the chunk is relevant to the query ('yes' or 'no'),
                            and the second element is the original text chunk.
    """
    chunk_document = [chunk[i : i + 512] for i in range(0, len(chunk), 512)]
    for chunk_data in chunk_document:
        for attempt in range(max_retries + 1):
            try:
                # Define the prompt
                content = [
                    f"system role: Given the user question: {query}, is the following text relevant and can be useful to "
                    f"answer the question?\n\n{chunk_data}\n\nAnswer 'yes' or 'no'."
                ]

                # Generate response using the Gemini model
                response = model.generate_content(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1024,
                        temperature=0.7,
                        top_p=0.9,
                    ),
                )
                relevance = response.text.strip().lower()

                # If response is "yes" or it's the last attempt, return the result
                if relevance == "yes":
                    return True, chunk
                elif attempt < max_retries:
                    time.sleep(1.0)
                else:
                    logging.info("Maximum retries reached; moving to next chunk.")
            except Exception as e:
                logging.error(f"Error analyzing chunk: {e}")
                if attempt < max_retries:
                    time.sleep(1.0)

    return False, chunk


def retrieve_documents(
    faiss_index: IndexFlatL2,
    model: SentenceTransformer,
    query: str,
    faiss_index_file_mapping: str = "/.faiss/faiss_index_file_mapping.json",
    model_name: str = "gemini-1.5-flash",
    k: int = 3,
    distance_threshold: float = None,
):
    """
    Retrieves relevant documents based on the user query using FAISS.

    Args:
        faiss_index: The FAISS index containing the indexed documents.
        model: The model (e.g., SentenceTransformer) used to encode the query.
        query (str): The user's query.
        faiss_index_file_mapping: A dictionary mapping FAISS indices to the actual document filenames.
        model_name (str, optional): The name of the generative AI model to load. Defaults to "gemini-1.5-flash".
        k (int): The number of documents to retrieve.
        distance_threshold (float, optional): Minimum distance threshold for relevant documents.

    Returns:
        list: A list of document filenames corresponding to the retrieved documents.
    """
    try:
        logging.info(f"Retrieving documents for query: {query}")

        # Load filenames mapping from the JSON file
        with open(faiss_index_file_mapping, "r") as f:
            filenames_mapping = json.load(f)

        # Encode the query into embeddings
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32")

        # Search the FAISS index
        distances, indices = faiss_index.search(query_embedding, k)

        files = []
        session_documents_folder = os.path.abspath(os.path.join("retrieved_documents"))
        os.makedirs(session_documents_folder, exist_ok=True)

        # Load the Gemini model
        model = load_model(model_name)

        # Iterate over the search results
        for dist, idx in zip(distances[0], indices[0]):
            # Skip documents that do not meet the distance threshold if provided
            if distance_threshold is not None and dist > distance_threshold:
                logging.info(
                    f"Document at index {idx} skipped due to distance threshold: {dist}"
                )
                continue

            # Fetch the document filename from the filenames_mapping based on the FAISS index
            doc_filename = filenames_mapping.get(str(idx), None)

            if doc_filename:
                # Path to the uploaded document
                doc_path = os.path.abspath(
                    os.path.join("uploaded_documents", doc_filename)
                )

                if os.path.exists(doc_path):
                    dest_path = os.path.join(
                        session_documents_folder,
                        f"{idx}_{doc_filename}",
                    )

                    # Delete if destination file already exists
                    if os.path.exists(dest_path):
                        os.remove(dest_path)

                    # Copy document to the static folder
                    with open(doc_path, "rb") as f:
                        content = f.read()

                        # Analyze if the document is relevant to query
                        is_relevant, useful_chunk = analyze_chunk_with_llm(
                            model=model, chunk=content, query=query
                        )
                        logging.info(
                            f"Is relevant status: {is_relevant} for document: {doc_path}"
                        )
                        if not is_relevant:
                            # Skip if the document is not relevant
                            continue

                        # Save the relevant chunk to the static folder
                        with open(dest_path, "wb") as f:
                            f.write(useful_chunk)
                        logging.info(f"Saved document: {dest_path}")

                    # Store the relative path from the static folder
                    files.append(dest_path)
                    logging.info(f"Added document to list: {dest_path}")
                else:
                    logging.warning(f"Document not found: {doc_path}")
            else:
                logging.warning(f"No filename found for index {idx}")

        logging.info(f"Total {len(files)} documents retrieved. Paths: {files}")
        return files
    except Exception as e:
        logging.exception(traceback.format_exc())
        logging.error(f"Error retrieving documents: {e}")
        return []
