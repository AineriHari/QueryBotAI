# Local LLM Project

## Overview

This project aims to create a local Large Language Model (LLM) for learning purposes. The LLM will utilize several tools to process documents, generate embeddings, and provide responses based on user queries.

## Tools and Technologies

### Faiss
A library for efficient similarity search and clustering of dense vectors. It will be used to store and index the document embeddings.

### SentenceTransformers
A Python library for generating sentence embeddings using state-of-the-art transformer models. It will convert documents into embeddings.

### Gemini Flash Pro
An API that will be used to execute queries and generate responses based on the content retrieved.

## Setup Instructions

### 1. Create a Virtual Environment
Run the following command to create a virtual environment and activate it:

```bash
python -m venv <env_name>
