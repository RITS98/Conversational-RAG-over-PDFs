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

