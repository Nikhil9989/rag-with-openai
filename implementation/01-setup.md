# Setup and Environment

This guide covers the initial setup and environment configuration for building a RAG system with OpenAI.

## Prerequisites

- Python 3.8+
- OpenAI API key
- Basic knowledge of Python and virtual environments

## Step 1: Create a Virtual Environment

```bash
# Create a new directory for your project
mkdir rag-project
cd rag-project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## Step 2: Install Required Packages

```bash
pip install openai langchain langchain-openai pypdf tiktoken chromadb pandas numpy streamlit
```

## Step 3: Set Up Environment Variables

Create a `.env` file in your project root directory:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

Create a Python script to load environment variables:

```python
# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

## Step 4: Initialize OpenAI Client

```python
# openai_client.py
import os
from openai import OpenAI
from config import OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Test the client
def test_openai_connection():
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=10
        )
        print("OpenAI connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error connecting to OpenAI: {e}")
        return False

if __name__ == "__main__":
    test_openai_connection()
```

## Step 5: Project Structure

Organize your project with the following structure:

```
rag-project/
├── .env                  # Environment variables
├── venv/                 # Virtual environment
├── config.py             # Configuration settings
├── openai_client.py      # OpenAI client initialization
├── data/                 # Directory for source documents
├── database/             # Directory for vector database
│   └── chroma/           # For Chroma DB files
├── document_processor.py # Document loading and processing
├── embeddings.py         # Embedding generation and storage
├── retriever.py          # Document retrieval mechanisms
├── generator.py          # Response generation
└── app.py                # Streamlit web application
```

## Step 6: Initialize Vector Database

```python
# embeddings.py
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from config import OPENAI_API_KEY

# Initialize embedding function
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

# Initialize Chroma vector database
def init_vector_db(persist_directory="./database/chroma"):
    vector_db = Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    return vector_db

if __name__ == "__main__":
    # Test vector database initialization
    db = init_vector_db()
    print("Vector database initialized successfully!")
```

## Next Steps

Now that you have set up your environment, you can proceed to the next guide: [Document Processing](02-document-processing.md).
