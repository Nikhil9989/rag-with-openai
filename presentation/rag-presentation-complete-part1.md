---
marp: true
theme: default
paginate: true
backgroundColor: #FFFFFF
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  section {
    font-family: 'Arial', sans-serif;
  }
  h1, h2, h3 {
    color: #1e88e5;
  }
  code {
    background: #f0f0f0;
    border-radius: 4px;
    padding: 2px 4px;
  }
  .highlight {
    color: #e91e63;
    font-weight: bold;
  }
  ul li, ol li {
    margin-bottom: 0.6em;
  }
  .small {
    font-size: 0.8em;
  }
  .center {
    text-align: center;
  }
---

<!-- Title Slide -->
# Building RAG with OpenAI
## A Comprehensive Guide for Beginners

<span class="small">
March 28, 2025
</span>

<!-- 
SPEAKER NOTES:

Good morning/afternoon everyone! Today I'll be presenting a comprehensive guide to building Retrieval-Augmented Generation systems with OpenAI.

This presentation is designed to take you through the entire process of creating a RAG system from scratch, even if you have no prior experience. By the end, you'll understand how RAG works, why it's valuable, and how to implement it with OpenAI's tools.

Let's get started!
-->

---

<!-- Agenda Slide -->
# Agenda

1. What is RAG?
2. Why Use RAG?
3. Key Components
4. Architecture Overview
5. Implementation Steps
6. Best Practices
7. Demo & Use Cases
8. Resources & Next Steps

<!-- 
SPEAKER NOTES:

Here's what we'll cover today:

First, I'll explain what RAG is and why it's becoming so important in modern AI applications.

Then, we'll dive into the key components that make up a RAG system and look at the overall architecture.

After that, we'll go through implementation steps with actual code examples.

Finally, we'll cover best practices, look at some use cases, and discuss resources for further learning.

Any questions about the agenda before we dive in?
-->

---

<!-- What is RAG Slide -->
# What is RAG?
## Retrieval-Augmented Generation

- **Definition**: A technique that combines retrieval systems with generative AI
- **Core Concept**: Enhance LLM outputs with relevant external knowledge
- **Process**:
  1. Store and index knowledge in a database
  2. Retrieve relevant information for a given query
  3. Augment LLM context with retrieved information
  4. Generate accurate, contextual responses

<!-- 
SPEAKER NOTES:

RAG stands for Retrieval-Augmented Generation. It's a hybrid approach that combines the power of large language models with information retrieval systems.

The core concept is simple yet powerful: we enhance LLM outputs by providing them with relevant external knowledge before they generate a response.

The process works in four steps:
1. First, we store and index our knowledge base in a vector database
2. When a user asks a question, we retrieve the most relevant information
3. We then augment the LLM's context with this retrieved information
4. Finally, the LLM generates a response based on both its training and the retrieved context

This approach addresses many limitations of using LLMs alone, as we'll see next.
-->

---

<!-- Why Use RAG Slide -->
# Why Use RAG?

<div class="columns">
<div>

- ✅ **Overcome knowledge cutoffs**
  - Access to latest information
  - Domain-specific knowledge

- ✅ **Reduce hallucinations**
  - Ground responses in facts
  - Verifiable information sources

- ✅ **Cost-efficiency**
  - Cheaper than fine-tuning
  - Scalable architecture
</div>
<div>

- ✅ **Data privacy**
  - Keep sensitive data in your control
  - Selective information sharing

- ✅ **Customization**
  - Tailor responses to your domain
  - Consistent organizational knowledge

- ✅ **Transparency**
  - Source attribution
  - Explainable responses
</div>
</div>

<!-- 
SPEAKER NOTES:

Why should we use RAG instead of just using a language model directly? There are several compelling benefits:

First, RAG helps overcome knowledge cutoffs. LLMs only know what they were trained on, but RAG can access the latest information and domain-specific knowledge.

Second, it significantly reduces hallucinations. By grounding responses in factual information from trusted sources, the model is less likely to make things up.

RAG is also cost-efficient compared to fine-tuning models, which can be expensive and time-consuming.

For enterprises, data privacy is crucial - RAG lets you keep sensitive data under your control.

It also enables customization to your specific domain and provides transparency through source attribution.

In our NetApp context, this means we can build AI systems that leverage our proprietary documentation and knowledge base while maintaining security and accuracy.
-->

---

<!-- Key Components - Document Processing Slide -->
# Key Components

## 1. Document Processing Pipeline

- **Document ingestion**
  - File formats (PDF, DOCX, HTML, etc.)
  - Web scraping capabilities
  - Database connectors

- **Text extraction & cleaning**
  - OCR for images and scanned documents
  - Structured data parsing
  - Noise removal and normalization

- **Chunking strategies**
  - Size-based chunking
  - Semantic chunking
  - Hierarchical chunking

<!-- 
SPEAKER NOTES:

Let's look at the key components of a RAG system, starting with the document processing pipeline.

The first step is document ingestion. Your RAG system needs to handle various file formats like PDFs, Word documents, and HTML. You might also need web scraping or database connectors depending on where your knowledge resides.

Next comes text extraction and cleaning. This might involve OCR for scanned documents, parsing structured data, and removing noise like headers, footers, and formatting artifacts.

One of the most critical aspects is chunking - dividing documents into smaller pieces. We typically use size-based chunking (breaking by token count), semantic chunking (preserving meaning), or hierarchical chunking (maintaining document structure).

The chunking strategy dramatically impacts retrieval quality. Too small chunks lose context; too large chunks dilute relevance. At NetApp, we've found that chunks of 800-1000 tokens with 200 token overlap work well for technical documentation.
-->

---

<!-- Key Components - Vector Database Slide -->
# Key Components

## 2. Vector Database

- **Embedding generation**
  - OpenAI embeddings (text-embedding-3-large)
  - Dimensionality considerations
  - Batch processing for efficiency

- **Vector storage options**
  - Pinecone, Weaviate, Qdrant, Chroma
  - Managed vs. self-hosted
  - Scaling considerations

- **Similarity search mechanisms**
  - Approximate nearest neighbors
  - Exact k-NN search
  - Performance tradeoffs

<!-- 
SPEAKER NOTES:

The second key component is the vector database, which is the heart of RAG's retrieval capabilities.

For embedding generation, we'll be using OpenAI's text-embedding-3-large model, which provides state-of-the-art semantic understanding. When implementing, we'll need to consider dimensionality and batch processing for efficiency.

For vector storage, we have several options: Pinecone, Weaviate, Qdrant, and Chroma are the most popular. Each has its strengths - Pinecone is fully managed but costs more, while Chroma is open-source and can be self-hosted.

The similarity search mechanism is what finds relevant content. Most vector databases use approximate nearest neighbors algorithms for speed, with tradeoffs between search accuracy and performance.

In our implementation, we'll use Chroma for simplicity, but the concepts apply across all vector databases. For production at scale, we'd likely use Pinecone or a self-hosted Weaviate instance.
-->

---

<!-- Key Components - Retrieval System Slide -->
# Key Components

## 3. Retrieval System

- **Query processing**
  - Query understanding
  - Intent classification
  - Semantic parsing

- **Advanced retrieval methods**
  - Hybrid search (keyword + semantic)
  - Query expansion
  - Dense passage retrieval
  - Maximum marginal relevance

- **Re-ranking algorithms**
  - Cross-encoder re-ranking
  - Contextual relevance scoring
  - Multi-stage retrieval pipelines

<!-- 
SPEAKER NOTES:

The third component is the retrieval system, which determines how we find relevant information.

Query processing is the first step, where we understand what the user is asking. This might involve intent classification or semantic parsing to optimize the search.

For retrieval, we've moved beyond simple semantic search. Advanced methods include hybrid search, which combines keyword and vector search; query expansion, which adds related terms; and techniques like maximum marginal relevance to ensure diverse results.

Re-ranking algorithms can further improve results by taking a first set of candidates and reordering them based on more sophisticated criteria.

In our implementation, we'll start with basic semantic search and then show how to add hybrid search for better results. The beauty of RAG is that we can improve each component independently - for instance, enhancing retrieval without changing the rest of the system.
-->

---

<!-- Key Components - Generation Slide -->
# Key Components

## 4. Generation with Context

- **Prompt engineering**
  - Context window optimization
  - Instruction clarity
  - Few-shot examples

- **Context integration**
  - Document assemblage
  - Citation frameworks
  - Token management

- **Response synthesis**
  - OpenAI model selection
  - Temperature settings
  - Output formatting

<!-- 
SPEAKER NOTES:

The final key component is the generation phase, where we use the LLM to create responses.

Prompt engineering is crucial here. We need to optimize how we use the context window, provide clear instructions to the LLM, and sometimes include examples of desired outputs.

Context integration focuses on how we combine the retrieved documents into a coherent context. This includes managing tokens to stay within model limits and setting up citation frameworks.

For response synthesis, we'll use OpenAI's models - typically GPT-4 for maximum quality or GPT-3.5 for cost efficiency. We'll also look at temperature settings, which control creativity versus determinism.

The generation component is where prompt engineering really shines. A well-crafted prompt can dramatically improve the quality and consistency of responses. We'll show specific prompts that work well for different RAG scenarios.
-->

---

<!-- Architecture Overview Slide -->
# Architecture Overview

```
┌────────────────┐    ┌─────────────────┐    ┌───────────────────┐
│ Document Store │───▶│ Vector Database │───▶│ Similarity Search │
└────────────────┘    └─────────────────┘    └───────────────────┘
        │                                              │
        ▼                                              ▼
┌────────────────┐                           ┌───────────────────┐
│ Text Chunks    │                           │ Relevant Chunks   │
└────────────────┘                           └───────────────────┘
                                                      │
┌────────────────┐                                    ▼
│ User Query     │───────────────────────────▶┌───────────────────┐
└────────────────┘                            │ Context Assembly  │
                                              └───────────────────┘
                                                      │
┌────────────────┐                                    ▼
│ LLM Response   │◀──────────────────────────┌───────────────────┐
└────────────────┘                           │ OpenAI Generation │
                                             └───────────────────┘
```

<!-- 
SPEAKER NOTES:

Here's a visualization of the complete RAG architecture we've been discussing.

The process starts at the top left with our document store. Documents are processed into text chunks and stored in a vector database.

When a user submits a query, the system performs similarity search to find relevant chunks from the vector database.

These chunks go through context assembly, where they're formatted and combined to create the context for the LLM.

Finally, OpenAI's models generate a response based on the query and assembled context.

The beauty of this architecture is its modularity - we can improve any component independently. For example, we could enhance the similarity search without changing the rest of the system.

This architecture is what we'll implement step by step in the next section, starting with the environment setup.
-->

---

<!-- Implementation Step 1 Slide -->
# Implementation Steps

## 1. Setup & Environment

```python
# Install required packages
!pip install openai langchain chromadb tiktoken

# Set OpenAI API key
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
client = OpenAI()

# Initialize vector database connection
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
vector_db = Chroma(embedding_function=embedding_function, persist_directory="./chroma_db")
```

<!-- 
SPEAKER NOTES:

Let's start implementing a RAG system, beginning with the setup and environment.

First, we install the required packages: OpenAI for API access, LangChain for the RAG framework, ChromaDB for vector storage, and tiktoken for token counting.

Next, we set the OpenAI API key. In production, you'd use environment variables or a secure secret management system.

Then we initialize the vector database connection. We're using Chroma with OpenAI's text-embedding-3-large model. The persist_directory parameter tells Chroma where to store its files.

This code establishes the foundation for our RAG system. Note that we're using LangChain, which is a popular framework that simplifies RAG implementation. However, you could also build this from scratch if needed.

For our team implementation, we'll need to ensure everyone has API access and the necessary permissions set up before we begin.
-->

---

<!-- Implementation Step 2 Slide -->
# Implementation Steps

## 2. Document Processing

```python
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = DirectoryLoader('./documents/', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
text_chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(text_chunks)} chunks")
```

<!-- 
SPEAKER NOTES:

Step 2 is document processing, where we load and chunk our documents.

We're using LangChain's document loaders - specifically PyPDFLoader for PDF files, wrapped in a DirectoryLoader to process all PDFs in a directory.

After loading, we split the documents into chunks using RecursiveCharacterTextSplitter. We've set a chunk size of 1000 characters with 200 characters of overlap between chunks. The overlap helps maintain context across chunk boundaries.

This recursive splitter is smart - it tries to split on logical boundaries like paragraphs and sentences before resorting to character-level splits.

For our NetApp documentation, we'll want to experiment with these parameters - larger chunks preserve more context but reduce precision, while smaller chunks improve precision but may lose context. The overlap helps balance these concerns.

We print a summary showing how many documents were processed and how many chunks were created.
-->

---

<!-- Implementation Step 3 Slide -->
# Implementation Steps

## 3. Vector Database Implementation

```python
# Create embeddings and store in vector database
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

# Store document chunks in vector database
vector_db = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)

# Persist to disk
vector_db.persist()
```

<!-- 
SPEAKER NOTES:

Step 3 focuses on implementing the vector database, where we'll store our document embeddings.

We first create an embedding function using OpenAI's text-embedding-3-large model, which converts text into high-dimensional vectors that capture semantic meaning.

Then we initialize Chroma and populate it with our document chunks. Chroma.from_documents handles both creating the embeddings and storing them in the database.

Finally, we call persist() to save the database to disk, allowing us to reuse it across sessions without reprocessing all documents.

This step might take some time, especially with large document collections, because each chunk needs to be converted to an embedding vector. For larger implementations, we'd want to batch this process and possibly use parallel processing.

OpenAI's embedding model costs about $0.0001 per 1,000 tokens, so embedding costs are typically quite low compared to generation costs.
-->
