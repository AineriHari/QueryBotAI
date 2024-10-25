"""
Module for indexing and querying documents using FAISS and a SentenceTransformer model.

This module provides functionality to upload documents, create a FAISS index for efficient
retrieval, and generate responses based on user queries. It includes command-line interface
support for indexing and querying operations.

Key Functions:
- create_filenames_mapping: Maps FAISS indices to actual document filenames.
- load_faiss_model: Loads the FAISS index and embedding model.
- upload_files: Handles file uploads and saves them to the designated folder.
- index_documents_for_files: Indexes uploaded documents using the FAISS index.
- query_documents: Retrieves documents based on a query.
- generate_response_for_query: Generates a response from retrieved documents.
- main: Command-line interface for the module.
"""

import os
import argparse
import faiss
from utils.indexer import index_documents
from utils.retriever import retrieve_documents
from utils.responder import generate_response
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure folders
UPLOAD_FOLDER = 'uploaded_documents'
STATIC_FOLDER = 'static'
INDEX_FOLDER = os.path.join(os.getcwd(), '.faiss')
FAISS_INDEX_FILE = os.path.join(INDEX_FOLDER, "index.faiss")

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Initialize global variable for the faiss model
faiss_model = None
model = None


def create_filenames_mapping(document_folder):
    """
    Creates a mapping from FAISS index to the actual document filenames.
    
    Args:
        document_folder: The folder where the documents are stored.
    
    Returns:
        dict: A dictionary mapping FAISS indices to filenames.
    """
    filenames_mapping = {}
    
    # List all files in the document folder
    document_files = os.listdir(document_folder)
    
    # Assuming documents are named based on their index in FAISS (0.pdf, 1.pdf, ...)
    for idx, doc_file in enumerate(document_files):
        # Map the FAISS index to the document filename
        filenames_mapping[idx] = doc_file
    
    return filenames_mapping


def load_faiss_model():
    """
    Loads the FAISS index and the SentenceTransformer model.

    This function checks if the FAISS index file exists and attempts to load both the FAISS
    index and the SentenceTransformer model. If loading fails, it prints an error message.
    """
    global faiss_model, model

    # Check if the FAISS index file exists
    if os.path.exists(FAISS_INDEX_FILE):
        try:

            # Set number of threads for faster operations
            faiss.omp_set_num_threads(8)

            # Load the FAISS index
            faiss_model = faiss.read_index(FAISS_INDEX_FILE)
            print("FAISS index loaded")

            # Load the embedding model (e.g., SentenceTransformer)
            model = SentenceTransformer(
                'sentence-transformers/paraphrase-MiniLM-L6-v2',
                local_files_only=True
            )
            print("SentenceTransformer model loaded")

        except Exception as e:
            print(f"Error loading FAISS index or model: {e}")
    else:
        print("No FAISS index found.")


def upload_files(files):
    """
    Uploads files to the designated upload folder.

    Args:
        files: A list of file paths to upload.

    Returns:
        list: A list of filenames of the uploaded files.
    """
    uploaded_files = []
    for file in files:
        filename = secure_filename(file)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file, 'r') as fr:
            with open(file_path, 'w') as fw:
                fw.write(fr.read())
        uploaded_files.append(filename)
        print(f"File saved: {file_path}")
    return uploaded_files


def index_documents_for_files(uploaded_files):
    """
    Indexes the uploaded documents using FAISS.

    Args:
        uploaded_files: A list of filenames that have been uploaded.

    Returns:
        list: A list of filenames that were successfully indexed.
    """
    global faiss_model
    if not uploaded_files:
        print("No files to index.")
        return []
    
    try:
        faiss_model = index_documents(UPLOAD_FOLDER, index_path=INDEX_FOLDER)
        print("Documents indexed successfully.")
        return uploaded_files
    except Exception as e:
        print(f"Error indexing documents: {str(e)}")
        return []


def query_documents(query):
    """
    Queries the FAISS index for relevant documents based on the input query.

    Args:
        query: The query string to search for.

    Returns:
        list: A list of documents retrieved from the FAISS index.
    """
    if faiss_model is None:
        print("faiss model not loaded.")
        return []

    # Create the mapping between FAISS indices and filenames
    filenames_mapping = create_filenames_mapping(UPLOAD_FOLDER)

    retrieved_documents = retrieve_documents(faiss_model, model, query, filenames_mapping)
    print(f"Retrieved documents: {retrieved_documents}")
    return retrieved_documents


def generate_response_for_query(query, retrieved_documents):
    """
    Generates a response based on the input query and retrieved documents.

    Args:
        query: The query string.
        retrieved_documents: The documents retrieved based on the query.

    Returns:
        str: The generated response, formatted as HTML.
    """
    try:
        response = generate_response(query=query, documents=retrieved_documents)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response."


def main():
    """
    Main function to run the command-line interface for indexing and querying documents.

    This function sets up argument parsing, loads the FAISS model, and handles the actions
    for either indexing documents or querying based on user input.
    """
    parser = argparse.ArgumentParser(description="Command-line tool for document indexing and querying.")
    parser.add_argument('--action', choices=['query', 'index'], required=True, help="Action to perform")
    parser.add_argument('--files', nargs='*', help="Files to upload for indexing")
    parser.add_argument('--query', help="Query to ask the faiss model")
    
    args = parser.parse_args()

    # Load faiss model when starting the script
    load_faiss_model()

    if args.action == 'query':
        if not args.query:
            print("Query is required.")
            return
        retrieved_documents = query_documents(args.query)
        response = generate_response_for_query(args.query, retrieved_documents)
        with open("response.txt", "w") as file:
            file.write(response)
        print(f"Generated response successfully.")

    elif args.action == 'index':
        if not args.files:
            print("Files are required for indexing.")
            return
        uploaded_files = upload_files(args.files)
        uploaded_files = index_documents_for_files(uploaded_files)
        print(f"Files uploaded and indexed: {uploaded_files}")


if __name__ == "__main__":
    main()
