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


def retrieve_documents(faiss_index, model, query, filenames_mapping, k=3):
    """
    Retrieves relevant documents based on the user query using FAISS.

    Args:
        faiss_index: The FAISS index containing the indexed documents.
        model: The model (e.g., SentenceTransformer) used to encode the query.
        query (str): The user's query.
        filenames_mapping: A dictionary mapping FAISS indices to the actual document filenames.
        k (int): The number of documents to retrieve.

    Returns:
        list: A list of document filenames corresponding to the retrieved documents.
    """
    try:
        print(f"Retrieving documents for query: {query}")
        
        # Encode the query into embeddings
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')

        # Search the FAISS index
        _, indices = faiss_index.search(query_embedding, k)
        
        files = []
        session_documents_folder = os.path.abspath(os.path.join('static', 'documents'))
        os.makedirs(session_documents_folder, exist_ok=True)
        
        # Iterate over the search results
        for idx in indices[0]:  # indices[0] gives the indices for the top k results
            # Fetch the document filename from the filenames_mapping based on the FAISS index
            doc_filename = filenames_mapping.get(idx, None)
            
            if doc_filename:
                # Path to the uploaded document
                doc_path = os.path.abspath(os.path.join('uploaded_documents', doc_filename))
                
                if os.path.exists(doc_path):
                    # Copy or move the document to the static folder
                    dest_path = os.path.join(session_documents_folder, f"retrieved_{idx}_{doc_filename}")
                    if not os.path.exists(dest_path):
                        # Copy document to the static folder
                        with open(doc_path, 'rb') as f:
                            content = f.read()
                        with open(dest_path, 'wb') as f:
                            f.write(content)
                        print(f"Saved document: {dest_path}")
                    
                    # Store the relative path from the static folder
                    relative_path = os.path.abspath(os.path.join('static', 'documents', f"retrieved_{idx}_{doc_filename}"))
                    files.append(relative_path)
                    print(f"Added document to list: {relative_path}")
                else:
                    print(f"Document not found: {doc_path}")
            else:
                print(f"No filename found for index {idx}")
        
        print(f"Total {len(files)} documents retrieved. Paths: {files}")
        return files
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []
