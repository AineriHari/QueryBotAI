"""
Module for retrieving documents based on user queries using FAISS.

This module provides functionality to retrieve relevant documents from a FAISS index
using embeddings generated from user queries. It handles document management by copying
retrieved documents to a designated folder for easy access.

Key Functions:
- retrieve_documents: Retrieves relevant documents for a given query using a FAISS index.
"""

import os
import numpy as np
import traceback
from faiss.swigfaiss import IndexFlatL2
import logging
import json
from sentence_transformers import SentenceTransformer


# Set up logging
logging.basicConfig(level=logging.INFO)


def retrieve_documents(
    faiss_index: IndexFlatL2,
    model: SentenceTransformer,
    query: str,
    faiss_index_file_mapping: str = "/.faiss/faiss_index_file_mapping.json",
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

                        # Save the relevant chunk to the static folder
                        with open(dest_path, "wb") as f:
                            f.write(content)
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
