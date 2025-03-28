# RAG Code Samples

This directory contains code samples for implementing Retrieval-Augmented Generation (RAG) with OpenAI.

## Files

- `rag_simple.py`: A simple, standalone RAG implementation
- `streamlit_app.py`: A web-based RAG application using Streamlit

## Usage Instructions

### Simple RAG Implementation

The `rag_simple.py` file provides a basic implementation that you can run from the command line:

1. Install the required packages:
   ```bash
   pip install openai langchain langchain-openai pypdf tiktoken chromadb
   ```

2. Set your OpenAI API key in the script or as an environment variable.

3. Modify the file path and query in the `__main__` section:
   ```python
   file_path = "path/to/your/document.pdf"  # Replace with your document path
   query = "What is RAG and how does it work?"  # Replace with your query
   ```

4. Run the script:
   ```bash
   python rag_simple.py
   ```

### Streamlit Web Application

The `streamlit_app.py` provides a user-friendly web interface:

1. Install the required packages:
   ```bash
   pip install openai langchain langchain-openai pypdf tiktoken chromadb streamlit
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. In the web interface:
   - Enter your OpenAI API key
   - Upload PDF documents
   - Configure chunking parameters
   - Process the documents
   - Start asking questions!

## Key Features

- Document loading and processing with LangChain
- Vector embeddings using OpenAI's embedding models
- Similarity search with Chroma vector database
- Context-based response generation with OpenAI's models
- User-friendly interface with Streamlit

## Customization

You can modify these examples to:
- Support different document types (DOCX, TXT, HTML, etc.)
- Use alternative embedding models
- Implement different vector databases (Pinecone, Weaviate, etc.)
- Add advanced retrieval mechanisms
- Customize the prompt templates

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in the installation steps
