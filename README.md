# QueryBotAI

This project aims to create a lightweight local Large Language Model (LLM) for document-based query search, code generation, and chatbot interaction. The system uses FAISS for efficient indexing, Sentence Transformers for embedding generation, and Gemini models for generative responses. The chatbot provides a conversational interface for users to query documents and interact with the LLM effectively.

## Features

### Document-Based Query Search: 
Retrieve the most relevant content from documents using FAISS and embeddings.

### Code Generation: 
Generate code snippets based on user input with minimal explanations.

### Interactive Chatbot: 
Provides a conversational interface for querying documents and interacting with the LLM.

## Tools and Technologies

### Faiss

A library for efficient similarity search and clustering of dense vectors. It will be used to store and index the document embeddings.

### SentenceTransformers

A Python library for generating sentence embeddings using state-of-the-art transformer models. It will convert documents into embeddings.

### Gemini Flash Pro

An API that will be used to execute queries and generate responses based on the content retrieved.

## Environment Setup

To set up your environment, follow these steps:

Note: Use Mac or Linux OS

1. **Create a Virtual Environment**  
   Run the following command to create a virtual environment and activate it:

   ```bash
   python -m venv <env_name>

   ```

2. **Activate the environment**

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
   SERVER_NAME=your-server-ip-or-localhost
   SERVER_PORT=your-server-port
   ```

5. **Run the Application**

   - Start the application by running:

   ```bash
   python chatbot.py
   ```

## Contributions

Feel free to open issues or submit pull requests if you'd like to contribute to this project.
