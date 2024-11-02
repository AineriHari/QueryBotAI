"""
Module for indexing and querying documents using FAISS and a SentenceTransformer model.

This module provides functionality to upload documents, create a FAISS index for efficient
retrieval, and generate responses based on user queries. It includes command-line interface
support for indexing and querying operations.

Key Functions:
- load_faiss_model: Loads the FAISS index and embedding model.
- upload_files: Handles file uploads and saves them to the designated folder.
- index_documents_for_files: Indexes uploaded documents using the FAISS index.
- query_documents: Retrieves documents based on a query.
- generate_response_for_query: Generates a response from retrieved documents.
- main: Command-line interface for the module.
"""

import os
import traceback
import shutil
import faiss
import logging
from datetime import datetime
from typing import List
from utils.indexer import index_documents
from utils.retriever import retrieve_documents
from utils.responder import generate_response
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from faiss.swigfaiss import IndexFlatL2
from dotenv import load_dotenv
from termcolor import colored


# Load environment variables from .env file
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure folders
UPLOAD_FOLDER = "uploaded_documents"
RETRIVE_FOLDER = "retrieved_documents"
INDEX_FOLDER = os.path.abspath(os.path.join(os.getcwd(), ".faiss"))
FAISS_INDEX_FILE = os.path.abspath(os.path.join(INDEX_FOLDER, "index.faiss"))
FAISS_INDEX_FILE_MAPPING = os.path.abspath(
    os.path.join(INDEX_FOLDER, "faiss_index_file_mapping.json")
)


# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRIVE_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Initialize global variable for the faiss model
faiss_model: IndexFlatL2 = None
model: SentenceTransformer = None


def load_faiss_model() -> None:
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
            print_decorative_box("FAISS index loaded")

            # Load the embedding model (e.g., SentenceTransformer)
            model = SentenceTransformer(
                "sentence-transformers/paraphrase-MiniLM-L6-v2", local_files_only=True
            )
            logging.info("SentenceTransformer model loaded")
        except Exception as e:
            logging.error(f"Error loading FAISS index or model: {e}")
    else:
        logging.warning("No FAISS index found.")


def upload_files(files: List) -> List:
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
        with open(file, "r") as fr:
            with open(file_path, "w") as fw:
                fw.write(fr.read())
        uploaded_files.append(filename)
        logging.info(f"File saved: {file_path}")
    return uploaded_files


def index_documents_for_files(uploaded_files: List) -> List:
    """
    Indexes the uploaded documents using FAISS.

    Args:
        uploaded_files: A list of filenames that have been uploaded.

    Returns:
        list: A list of filenames that were successfully indexed.
    """
    global faiss_model
    if not uploaded_files:
        logging.error("No files to index.")
        return []

    try:
        faiss_model = index_documents(UPLOAD_FOLDER, index_path=INDEX_FOLDER)
        logging.info("FAISS Index Model loaded")
        print_decorative_box("Documents indexed successfully")
        return uploaded_files
    except Exception as e:
        logging.error(f"Error indexing documents: {str(e)}")
        return []


def query_documents(query: str) -> List:
    """
    Queries the FAISS index for relevant documents based on the input query.

    Args:
        query: The query string to search for.

    Returns:
        list: A list of documents retrieved from the FAISS index.
    """
    if faiss_model is None:
        logging.error("faiss model not loaded.")
        return []

    # Create the mapping between FAISS indices and filenames
    retrieved_documents = retrieve_documents(
        faiss_index=faiss_model,
        model=model,
        query=query,
        faiss_index_file_mapping=FAISS_INDEX_FILE_MAPPING,
    )
    logging.info(f"Retrieved documents: {retrieved_documents}")
    return retrieved_documents


def generate_response_for_query(
    query: str, retrieved_documents: List, search_type: str
) -> str:
    """
    Generates a response based on the input query and retrieved documents.

    Args:
        query: The query string.
        retrieved_documents: The documents retrieved based on the query.

    Returns:
        str: The generated response, formatted as HTML.
    """
    try:
        response = generate_response(
            query=query,
            documents=retrieved_documents,
            model_name=(
                os.getenv("MODEL_NAME")
                if os.getenv("MODEL_NAME", None)
                else "gemini-1.5-flash"
            ),
            search_type=search_type,
        )
        return response
    except Exception as e:
        logging.exception(traceback.format_exc())
        logging.error(f"Error generating response: {e}")
        return ""


def print_decorative_box(text: str) -> None:
    """
    Prints a decorative box around a given text.

    Args:
        text (str): The text to display inside the decorative box.

    Returns:
        None
    """
    box_length = len(text) + 4
    border = "+" * box_length
    middle = f"|  {text.center(len(text))}  |"
    logging.info(border)
    logging.info(middle)
    logging.info(border)


def cleanup(directory_paths: List) -> None:
    """
    Deletes all files and directories within the given directory paths.

    Args:
        directory_paths (List[str]): List of directory paths to clean up.

    Returns:
        None

    Note:
        If the specified directories do not exist, a message is printed.
    """
    for directory_path in directory_paths:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and os.path.exists(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            logging.info(
                f"Directory '{filename}' and all its contents have been removed."
            )
        else:
            logging.warning(f"Directory '{directory_path}' content does not exist.")


def _load_LLM_perform_query(query: str, search_type: str):
    """
    Loads the FAISS model, processes a given query, and generates a response based on the selected search type.

    This function verifies if a query is provided, retrieves relevant documents, and generates a response
    using the specified search type (e.g., text generation or code generation). It then logs the query and
    response details to a file named 'Chat_History.md' with a timestamp.

    Args:
        query (str): The query string to search and generate a response for.
        search_type (str): The type of search to be performed, either 'text-generation' or 'code-generation'.

    Returns:
        None
    """
    # Load faiss model when starting the script
    load_faiss_model()

    # verify the prompt query
    if not query:
        print_decorative_box("No query found in prompt template!!!")
    else:
        retrieved_documents = query_documents(query)
        response = generate_response_for_query(query, retrieved_documents, search_type)
        query_data = f"\n**Query (at {datetime.now().strftime('%d %b %Y, %-I %p %M Secs')}): {query}**\n"
        response = query_data + response + "\n" + "+" * 100 + "\n"
        with open("Chat_History.md", "a") as file:
            # write a query for reference
            file.write(response)
        print_decorative_box("Generated response successfully.")


def _select_search_type(search_type: str) -> str:
    """
    Determines the appropriate search type based on user input and logs the selection.

    This function interprets the search type input (either '0' for text generation or '1' for code generation).
    If an invalid option is provided, it raises a ValueError.

    Args:
        search_type (str): A string representation of the search type, where '0' represents text generation
                           and '1' represents code generation.

    Returns:
        str: The resolved search type, either 'text-generation' or 'code-generation'.

    Raises:
        ValueError: If an invalid search type is provided.
    """
    # select the search type
    if int(search_type.lower()[0]) == 0:
        search_type = "text-generation"
    elif int(search_type.lower()[0]) == 1:
        search_type = "code-generation"
    else:
        raise ValueError(
            f"You have selected wrong option, Your choice: {search_type}. please try again later!!!"
        )
    logging.info(f"You have selected search type: {search_type}")

    return search_type


def main() -> None:
    """
    Main function to run the command-line interface for indexing and querying documents.

    This function sets up argument parsing, loads the FAISS model, and handles the actions
    for either indexing documents or querying based on user input. If indexing is selected,
    it performs a cleanup of specified directories and uploads files. If querying, it loads
    the FAISS model, reads the query from a file, and generates a response.

    Returns:
        None
    """
    logging.info("Hello! Welcome to the local LLM...\n")
    logging.info(
        "This is a simple, fast-response document search LLM."
        "\nPlease start by indexing the documents you want to embed. You can then proceed with querying."
    )

    indexed = False

    while True:
        try:
            if not indexed:
                action = input(
                    colored(
                        f"\nPlease select an action:\n1. Index\n2. Query\nYour choice: ",
                        "red",
                    )
                )[0]
            else:
                action = "2"

            if action == "1":
                logging.info("Performing clean up action")
                # remove the uploaded_documents and static folder
                directory_paths = [
                    os.path.join(os.getcwd(), "uploaded_documents"),
                    os.path.join(os.getcwd(), "retrieved_documents"),
                ]
                cleanup(directory_paths)
                logging.info("Clean up process completed successfully")

                files = input(
                    colored("Provide the files by separating spaces\nFiles: ", "yellow")
                ).split(" ")

                uploaded_files = upload_files(files)
                uploaded_files = index_documents_for_files(uploaded_files)
                logging.info(f"Files uploaded and indexed: {uploaded_files}")
                indexed = True

            # if the action is "2", don't show the Index and Query action
            if action == "2":
                indexed = True

            # Generate response
            query = input(
                colored("Enter your next query, or type 'quit' to exit: ", "red")
            )
            if query.strip().lower() == "quit":
                # exit the loop
                logging.info("Thanks for using!!! Exiting the prompt.")
                break
            search_type = input(
                colored(
                    "Select your search type: text-generation(0)/code-generation(1) ?\n Your Choice: ",
                    "yellow",
                )
            )[0]

            # select the search type
            search_type = _select_search_type(search_type)

            # Load the LLM to perform query
            _load_LLM_perform_query(query=query, search_type=search_type)
        except Exception as exc:
            logging.error(traceback.format_exc())
            print_decorative_box(f"Failed!!!! {exc}")


if __name__ == "__main__":
    main()
