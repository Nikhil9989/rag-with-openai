# Simple RAG implementation with OpenAI and Chroma
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
client = openai.OpenAI()

# Document loading and processing
def load_and_process_documents(file_path, chunk_size=1000, chunk_overlap=200):
    # Load PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks

# Create and store embeddings
def create_vector_db(chunks, persist_directory="./chroma_db"):
    # Initialize embedding function
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create vector store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    
    # Persist to disk
    vector_db.persist()
    print(f"Created vector database with {len(chunks)} chunks")
    
    return vector_db

# Retrieve relevant documents
def retrieve_documents(vector_db, query, k=5):
    documents = vector_db.similarity_search(query, k=k)
    return documents

# Format context for the LLM
def format_context(documents):
    context = ""
    for i, doc in enumerate(documents):
        context += f"\nDocument {i+1}:\n{doc.page_content}\n"
    return context

# Generate response using OpenAI
def generate_response(query, context):
    prompt = f"""Answer the following question based ONLY on the provided context:

Context:
{context}

Question: {query}

Provide a comprehensive answer. If the information to answer the question is not present in the context, respond with "I don't have enough information to answer this question."
"""
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Main RAG pipeline function
def rag_pipeline(file_path, query):
    # Load and process documents
    chunks = load_and_process_documents(file_path)
    
    # Create vector database
    vector_db = create_vector_db(chunks)
    
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(vector_db, query)
    
    # Format context
    context = format_context(retrieved_docs)
    
    # Generate response
    answer = generate_response(query, context)
    
    return answer, retrieved_docs

# Example usage
if __name__ == "__main__":
    file_path = "path/to/your/document.pdf"  # Replace with your document path
    query = "What is RAG and how does it work?"  # Replace with your query
    
    answer, sources = rag_pipeline(file_path, query)
    
    print("Question:")
    print(query)
    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    for i, source in enumerate(sources):
        print(f"Source {i+1}:")
        print(source.page_content[:200] + "...")
        print()
