# Streamlit web app for RAG system
import os
import streamlit as st
import openai
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Set page configuration
st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")

# Initialize session state variables
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Title and description
st.title("📚 RAG Document Assistant")
st.markdown("""
This application uses Retrieval-Augmented Generation (RAG) to answer questions about your documents.
Upload PDF files, process them, and then ask questions to get accurate answers based on your documents.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        openai_client = openai.OpenAI(api_key=api_key)
        st.session_state.api_key_set = True
    
    # Document upload
    uploaded_files = st.file_uploader("Upload documents", type=["pdf"], accept_multiple_files=True)
    
    # Chunk size and overlap settings
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100,
                         help="Size of text chunks for processing")
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50,
                            help="Overlap between consecutive chunks")
    
    # Process documents button
    process_btn = st.button("Process Documents", disabled=not (uploaded_files and st.session_state.api_key_set))

# Main function to process documents
def process_documents(files, chunk_size, chunk_overlap):
    # Save uploaded files temporarily
    temp_dir = "./temp_docs/"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file in files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.getbuffer())
    
    # Load documents
    loader = DirectoryLoader(temp_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store in vector database
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    
    return vector_db, len(documents), len(text_chunks)

# Process documents when button is clicked
if process_btn and uploaded_files:
    with st.spinner("Processing documents... This might take a while depending on the size and number of documents."):
        try:
            st.session_state.vector_db, num_docs, num_chunks = process_documents(
                uploaded_files, chunk_size, chunk_overlap)
            st.session_state.documents_processed = True
            st.sidebar.success(f"Successfully processed {num_docs} documents into {num_chunks} chunks!")
        except Exception as e:
            st.sidebar.error(f"Error processing documents: {e}")

# Main content area for querying
if st.session_state.documents_processed and st.session_state.vector_db:
    st.header("Ask Questions About Your Documents")
    
    # Query input
    query = st.text_input("Enter your question:")
    num_results = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)
    
    if st.button("Submit", disabled=not query):
        with st.spinner("Generating answer..."):
            try:
                # Retrieve relevant documents
                retrieved_docs = st.session_state.vector_db.similarity_search(query, k=num_results)
                
                # Format context
                context = ""
                for i, doc in enumerate(retrieved_docs):
                    context += f"\nDocument {i+1}:\n{doc.page_content}\n"
                
                # Create prompt
                prompt = f"""Answer the following question based ONLY on the provided context:

Context:
{context}

Question: {query}

Provide a comprehensive answer. If the information to answer the question is not present in the context, respond with "I don't have enough information to answer this question."
"""
                
                # Generate response
                response = openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                answer = response.choices[0].message.content
                
                # Display the response
                st.subheader("Answer:")
                st.write(answer)
                
                # Display retrieved sources
                with st.expander("View Sources"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    # Display instructions when no documents are processed
    st.info("👈 Please upload PDF documents and process them using the sidebar controls.")
    
    # Example questions section
    st.header("How It Works")
    st.markdown("""
    ### Steps:
    1. **Upload Documents**: Add PDF files containing your information.
    2. **Process Documents**: Convert documents into searchable chunks.
    3. **Ask Questions**: Query your documents to get accurate answers.
    
    ### Benefits of RAG:
    - Get answers specific to your documents
    - Reduce hallucinations in AI responses
    - Maintain context across complex documents
    - Preserve privacy by using your own data
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
<p>Built with Streamlit, LangChain, and OpenAI</p>
</div>
""", unsafe_allow_html=True)
