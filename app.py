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
import gradio as gr
from datetime import datetime
from typing import List, Generator
from utils.indexer import index_documents
from utils.retriever import retrieve_documents
from utils.responder import generate_response
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from faiss.swigfaiss import IndexFlatL2
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment variables from .env file
load_dotenv(override=True)

# set the server name and port
SERVER_NAME = os.getenv("SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", 7860))

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
        with open(file, "rb") as fr:
            with open(file_path, "wb") as fw:
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
        search_type: This refers the what type of model should generate the response (Text Generation, Code Generation)
    Returns:
        str: The generated response, formatted as HTML.
    """
    try:
        response = generate_response(
            embedding_model=model,
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


def _load_LLM_perform_query(query: str, search_type: str) -> None:
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
        if not response.strip():
            return "Failed to generate the response. check the logs manually."
        query_data = f"\n**Query (at {datetime.now().strftime('%d %b %Y, %-I %p %M Secs')}): {query}**\n"
        response = query_data + response + "\n"
        with open("Chat_History.md", "a") as file:
            # write a query for reference
            file.write(response)
        print_decorative_box("Generated response successfully.")

        return response


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
    if search_type == 0:
        search_type = "text-generation"
    elif search_type == 1:
        search_type = "code-generation"
    else:
        raise ValueError(
            f"You have selected wrong option, Your choice: {search_type}. please try again later!!!"
        )
    logging.info(f"You have selected search type: {search_type}")

    return search_type


def load_and_index_file(files: List) -> str:
    """
    Uploads, cleans up, and indexes the provided files.

    This function removes temporary folders used for document storage, uploads the specified files,
    and indexes them for retrieval purposes. Once indexed, it returns a message listing the indexed
    files or an error message if the indexing process fails.

    Parameters:
    files (List): A list of file path to be uploaded and indexed.

    Returns:
    str: A message indicating the status of the indexing process.
    """

    if files is None:
        return "No file is uploaded for index"
    logging.info("Performing clean up section")

    # remove the uploaded_documents and static folder
    directory_paths = [
        os.path.join(os.getcwd(), "uploaded_documents"),
        os.path.join(os.getcwd(), "retrieved_documents"),
    ]
    cleanup(directory_paths)
    logging.info("Clean up process completed successfully")

    # upload files
    uploaded_files = upload_files(files)
    indexed_files = index_documents_for_files(uploaded_files)
    if indexed_files:
        return f"Indexed files: {', '.join(indexed_files)}"
    return "Failed to index files."


def query_response(query: str, search_type: int) -> str:
    """
    Processes a query to generate a response based on the specified search type.

    This function selects the appropriate search type (e.g., Text Generation or Code Generation),
    processes the query using an LLM model, and returns the generated response.

    Parameters:
    query (str): The user query for which a response is to be generated.
    search_type (str): The type of search to be performed (e.g., "Text Generation", "Code Generation").

    Returns:
    str: The generated response to the query.
    """
    search_type_selected = _select_search_type(search_type)
    response = _load_LLM_perform_query(
        query=query,
        search_type=search_type_selected,
    )
    return response


def generate_chat_bot(query: str) -> Generator:
    """
    Generates a response to a user's query using a language model.

    This function loads a language model based on an environment variable `MODEL_NAME`
    (or defaults to `"gemini-1.5-flash"` if the variable is not set) and generates a response
    to the input query. The response is generated incrementally and yields progressively
    more complete text for real-time display.

    Parameters:
        query (str): The user's input query to the chatbot.

    Yields:
        str: The progressively built response text, chunk by chunk, allowing for streaming output.
    """
    model_name = (
        os.getenv("MODEL_NAME") if os.getenv("MODEL_NAME", None) else "gemini-1.5-flash"
    )
    model = load_model(model_name)
    full_response = ""
    response = model.generate_content(
        query,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
        ),
        stream=True,
    )
    for chunk in response:
        full_response += chunk.text
        yield full_response


def get_readme_content():
    """
    Reads the content of the README.md file.

    Returns:
        str: The content of the README.md file. If the file is not found, returns a message indicating that.
    """
    try:
        return open("README.md", "r").read()
    except FileNotFoundError:
        return "README.md file not found."


# Function to load the Chat History content
def get_chat_history_content():
    """
    Reads the content of the Chat_History.md file.

    Returns:
        str: The content of the Chat_History.md file. If the file is not found, returns a message indicating that.
    """
    try:
        return open("Chat_History.md", "r").read()
    except FileNotFoundError:
        return "Chat_History.md file not found."


def main():
    """
    Initializes and launches the local LLM application with Gradio interfaces for file indexing and querying.

    This function sets up a Gradio interface for uploading and indexing documents, another for querying the indexed
    documents, and combines them into a tabbed Gradio applocation. It also loads the necessary models for indexing
    and querying before starting the app. Any exceptions during setup or launch are logged and displayed in the console.

    Returns:
    None
    """
    try:
        logging.info("Hello! Welcome to the local LLM...")

        # welcome page
        with gr.Blocks() as app:
            with gr.Tabs() as main_interface:
                with gr.Tab("Index Documents"):
                    with gr.Row():
                        # Left frame for prompt input
                        with gr.Column():
                            upload_input = gr.File(
                                label="Upload Documents", file_count="multiple"
                            )
                            upload_button = gr.Button("Upload")

                        # Right frame for response display
                        with gr.Column():
                            content_display = gr.Markdown()
                    # Click action to show "Loading..." message
                    upload_button.click(
                        fn=lambda q: """
                            <div class="loader"></div>
                            <style>
                                .loader {
                                    border: 8px solid #f3f3f3;
                                    border-top: 8px solid #3498db;
                                    border-radius: 50%;
                                    width: 100px;
                                    height: 100px;
                                    animation: spin 2s linear infinite;
                                    margin: 20px auto;
                                }

                                @keyframes spin {
                                    0% { transform: rotate(0deg); }
                                    100% { transform: rotate(360deg); }
                                }
                            </style>
                            """,
                        inputs=[upload_input],
                        outputs=content_display,
                        queue=True,
                    )

                    # Button click action to update the response display
                    upload_button.click(
                        fn=load_and_index_file,
                        inputs=[upload_input],
                        outputs=content_display,
                        show_progress=True,
                    )

                # Query Tab
                with gr.Tab("Prompt"):
                    with gr.Row():
                        # Left frame for prompt input
                        with gr.Column():
                            query_input = gr.Textbox(
                                label="Query", placeholder="Enter your query"
                            )
                            search_type = gr.Radio(
                                ["Text Generation", "Code Generation"],
                                label="Search Type",
                                type="index",
                                value="Text Generation",
                            )
                            query_button = gr.Button("Submit Query")

                        # Right frame for response display
                        with gr.Column():
                            query_response_display = gr.Markdown(
                                label="Response",
                                elem_id="response_display",
                            )

                    # Click action to show "Loading..." message
                    query_button.click(
                        fn=lambda q, s: """
                            <div class="loader"></div>
                            <style>
                                .loader {
                                    border: 8px solid #f3f3f3;
                                    border-top: 8px solid #3498db;
                                    border-radius: 50%;
                                    width: 100px;
                                    height: 100px;
                                    animation: spin 2s linear infinite;
                                    margin: 20px auto;
                                }

                                @keyframes spin {
                                    0% { transform: rotate(0deg); }
                                    100% { transform: rotate(360deg); }
                                }
                            </style>
                            """,
                        inputs=[query_input, search_type],
                        outputs=query_response_display,
                        queue=True,
                    )

                    # Button click action to update the response display
                    query_button.click(
                        fn=query_response,
                        inputs=[query_input, search_type],
                        outputs=query_response_display,
                        show_progress=True,
                    )

                # Chat bot
                with gr.Tab("Chat"):
                    with gr.Row():
                        # Left frame for prompt input
                        with gr.Column():
                            query_input = gr.Textbox(
                                label="Query", placeholder="Enter your query"
                            )
                            query_button = gr.Button("Submit Query")

                        # Right frame for response display
                        with gr.Column():
                            query_response_display = gr.Markdown(
                                label="Response",
                                elem_id="response_display",
                            )

                    # Click action to show "Loading..." message
                    query_button.click(
                        fn=lambda q: """
                            <div class="loader"></div>
                            <style>
                                .loader {
                                    border: 8px solid #f3f3f3;
                                    border-top: 8px solid #3498db;
                                    border-radius: 50%;
                                    width: 100px;
                                    height: 100px;
                                    animation: spin 2s linear infinite;
                                    margin: 20px auto;
                                }

                                @keyframes spin {
                                    0% { transform: rotate(0deg); }
                                    100% { transform: rotate(360deg); }
                                }
                            </style>
                            """,
                        inputs=[query_input],
                        outputs=query_response_display,
                        queue=True,
                    )

                    # Button click action to update the response display
                    query_button.click(
                        fn=generate_chat_bot,
                        inputs=[query_input],
                        outputs=query_response_display,
                        show_progress=True,
                    )
                    
                # New Tab for Readme and Chat History
                with gr.Tab("Readme and Chat History"):
                    # Display Readme Content
                    with gr.Row():
                        with gr.Column():  # Readme and Chat History buttons in one column
                            readme_button = gr.Button("Readme")
                            chat_history_button = gr.Button("Chat History")

                        with gr.Column():  # Display the content
                            content_display = gr.Markdown(
                                label="Response",
                                elem_id="response_display",
                            )

                    # Button click actions to display README or Chat History content
                    readme_button.click(
                        fn=get_readme_content,
                        inputs=None,
                        outputs=content_display,
                    )

                    chat_history_button.click(
                        fn=get_chat_history_content,
                        inputs=None,
                        outputs=content_display,
                    )

        # Load FAISS model and embedding model before starting the interface
        load_faiss_model()
        app.title = "localLLM"
        app.launch(server_name=SERVER_NAME, server_port=SERVER_PORT)
    except Exception as exc:
        logging.error(traceback.format_exc())
        print_decorative_box(f"Failed!!! {exc}")


if __name__ == "__main__":
    main()
