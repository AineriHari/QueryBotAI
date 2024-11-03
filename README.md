# Local LLM Project

## Overview

This project aims to create a lightweight local Large Language Model (LLM) for document based query searchs and code generation. The LLM will utilize several tools to process documents, generate embeddings, and provide responses based on user queries.

## Tools and Technologies

### Faiss

A library for efficient similarity search and clustering of dense vectors. It will be used to store and index the document embeddings.

### SentenceTransformers

A Python library for generating sentence embeddings using state-of-the-art transformer models. It will convert documents into embeddings.

### Gemini Flash Pro

An API that will be used to execute queries and generate responses based on the content retrieved.

## Environment Setup

To set up your environment, follow these steps:

1. **Create a Virtual Environment**  
   Run the following command to create a virtual environment and activate it:

   ```bash
   python -m venv <env_name>

   ```

2. **Activate the environment**

   - On Windows:

   ```bash
   <env_name>\Scripts\activate
   ```

   - On macOS/Linux:

   ```bash
   source <env_name>/bin/activate

   ```

3. **Install Required Packages**
   Install the necessary dependencies by running:

   ```bash
   pip install -r requirements.txt

   ```

4. **Configure Google Gemini API Key**

   - Create a Google Gemini API Key if you haven’t already.
   - Add your API key to the .env file with the following variables:

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   MODEL_NAME=your_model_name_here

   ```

5. **Run the Application**

   - Start the application by running:

   ```bash
   python app.py
   ```

   - You will be prompted to choose between two options:

   1. Index: Uploads documents for embedding generation. This will prepare the documents for querying.
   2. Query: Allows you to perform searches and get responses based on indexed content.

## Workflow

### 1. Indexing Documents

The indexing process involves several key steps to convert documents into embeddings and store them in the Faiss index. Follow the steps below to successfully index your documents:

#### Step 1: File Selection

- Upon running the application, you will be prompted to select the files you wish to index.
- You can specify multiple files, including text documents, PDFs, or any other supported formats.

#### Step 2: Document Processing

- The selected documents will be read and pre-processed to ensure they are in a suitable format for embedding.
- This may include text extraction from various file types and cleaning up the text for better results.

#### Step 3: Generating Embeddings

- Using the **SentenceTransformers** library, each document will be converted into a numerical representation (embedding).
- This step involves loading a pre-trained transformer model suitable for your domain and generating embeddings for each document.

#### Step 4: Storing Embeddings

- The generated embeddings will be indexed using **Faiss** for efficient similarity search.
- Each embedding will be associated with its corresponding document to ensure accurate retrieval later on.

#### Step 5: Confirmation

- After the indexing process is complete, you will receive a confirmation message indicating the successful indexing of your documents.
- You can then proceed to the query phase, where you can search for information based on the indexed documents.

### Example Code Snippet

Here’s a brief example of how you might implement the indexing step in Python:

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize Sentence Transformer
model = SentenceTransformer('model_name')

# Function to index documents
def index_documents(file_paths):
    embeddings = []
    for file_path in file_paths:
        # Read and process the document (pseudo-code)
        document_text = read_document(file_path)  # Implement this function
        # Generate embedding
        embedding = model.encode(document_text)
        embeddings.append(embedding)

    # Convert embeddings to a numpy array
    embeddings = np.array(embeddings).astype('float32')

    # Create a Faiss index and add embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("Documents indexed successfully!")

# Example usage
file_paths = ["doc1.txt", "doc2.pdf"]
index_documents(file_paths)
```
