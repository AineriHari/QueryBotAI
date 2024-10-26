Local LLM Project
Overview
This project aims to create a local Large Language Model (LLM) for learning purposes. The LLM will utilize several tools to process documents, generate embeddings, and provide responses based on user queries.

Tools and Technologies
Faiss: A library for efficient similarity search and clustering of dense vectors. It will be used to store and index the document embeddings.
SentenceTransformers: A Python library for generating sentence embeddings using state-of-the-art transformer models. It will convert documents into embeddings.
Gemini Flash Pro: An API that will be used to execute queries and generate responses based on the content retrieved.
Workflow
1. Indexing Documents
The LLM will prompt the user to specify the files they wish to index.
The specified documents will be converted into embeddings using SentenceTransformers.
These embeddings will be stored in a Faiss index for efficient retrieval.
2. Performing Queries
Users can input queries that will also be converted into embeddings.
The LLM will utilize Faiss to search for related documents based on the query embedding.
Once relevant documents are found, their content will be prepared along with the query for further processing.
3. Generating Responses
The prepared content and query will be executed using the Gemini Flash Pro API to generate responses.
The response will be based on the context provided by the related documents and the initial query.
Getting Started
Prerequisites
Python 3.x (Better to use latest python version)
Required libraries:
faiss
sentence-transformers
gemini-flash-pro (or any relevant package for the API)
Installation
You can install the required libraries using pip:

bash
Copy code
pip install -r requirement.txt
Usage
Run the main script to start the LLM.
Follow the prompts to index your documents.
Enter queries to retrieve information and generate responses.
