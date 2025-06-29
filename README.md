# Conversational RAG over PDFs

An interactive Streamlit app enabling **Conversational Retrieval-Augmented Generation (RAG)** over multiple uploaded PDF documents.  
Users upload PDFs, which are processed and embedded into a vector store. The app then supports chat-based Q&A using a Groq large language model, incorporating chat history and contextual document retrieval.

---

## Table of Contents

- [Demo](#demo)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Installation](#installation)  
- [Environment Setup](#environment-setup)  
- [Usage](#usage)  
- [Configuration](#configuration)  
- [Code Structure](#code-structure)  
- [Extending the Project](#extending-the-project)  
- [Troubleshooting](#troubleshooting)  
- [License](#license)  

---

## Demo

> _Coming soon!_ (Optionally, add a hosted demo link here.)

---

## Features

- Upload one or more PDF documents for conversational querying  
- Extracts, splits, and embeds PDF content into a persistent vector store using LangChain and Chroma  
- Uses **Groq's Gemma2-9b-It** LLM for context-aware Q&A with chat history support  
- Interactive chat interface with conversation memory persistence per session  
- Sidebar PDF previews with extracted text snippets  
- Modular, extensible, and production-ready codebase with optional LangSmith tracing for observability  

---

## Architecture

```mermaid
flowchart LR
    User((User))
    UI[Streamlit UI]
    PDFUpload[PDF Upload & Preprocessing]
    TextSplitter[RecursiveCharacterTextSplitter]
    Embeddings[OllamaEmbeddings]
    VectorStore[Chroma Vector Store]
    Retriever[History-aware Retriever]
    LLM[Groq Gemma2-9b-It LLM]
    ChatHistory[ChatMessageHistory]
    RAGChain[Retriever-Augmented Generation Chain]

    User --> UI
    UI --> PDFUpload
    PDFUpload --> TextSplitter
    TextSplitter --> Embeddings
    Embeddings --> VectorStore
    VectorStore --> Retriever
    Retriever --> RAGChain
    ChatHistory --> RAGChain
    LLM --> RAGChain
    UI --> RAGChain
````

## Installation

Requirements:  
- Python 3.8+

Poetry or pip for dependency management

Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Setup
Create a .env file in the project root with the following variables:

```plaintext
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # optional
LANGSMITH_PROJECT=your_project_name            # optional
LANGSMITH_RUN_NAME=rag_pdf_chat                 # optional
```

- GROQ_API_KEY: Required for Groq LLM access.

- HF_TOKEN: Required for HuggingFace models (used internally by embeddings).
- You can also use Ollama Vector Embeddings by setting the `OllamaEmbeddings` class in the code.

- LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_RUN_NAME: Optional, for LangSmith tracing.


## Usage
Run the app locally using:
```
streamlit run app.py
```

- Upload one or more PDF files using the uploader on the main page.

- View PDF previews and extracted text in the sidebar.

- Enter your questions in the input box to chat with the content of your PDFs.

- Session-based chat history is preserved and can be viewed/expanded below the chat.

- Change the session ID to start a new conversation context.


## Configuration
- Session Management: Controlled via session_id text input to isolate chat histories.

- Vector Store Persistence: Stores embeddings locally under ./chroma_store for faster repeated queries.

- Chunk Size / Overlap: Currently set to 5000 tokens with 500 overlap for text splitting; adjust in build_rag_chain.

- Model Selection: LLM (Gemma2-9b-It) and Embeddings (llama3.2) are configured but can be replaced with alternatives.

## Extending the Project
- Add summarization of extracted text snippets using the LLM for quick overviews.

- Add download buttons for PDF previews or extracted text summaries.

- Implement user authentication for multi-user deployment.

- Integrate database for long-term chat history persistence.

- Add Dockerfile for containerized deployment.

- Enhance PDF processing with OCR support for scanned documents.

- Experiment with other embedding models or retrievers (FAISS, Weaviate).

- Use LangSmith fully for LLM trace visualization and debugging.





