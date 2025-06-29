# Conversational RAG over PDFs with Optional Add-ons

import os
import streamlit as st
from dotenv import load_dotenv
import base64

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq


# --- Optional LangSmith Tracing Setup ---
def setup_langsmith_tracing():
    """
    Setup environment variables for LangSmith tracing if available.
    This allows capturing LLM interactions for monitoring/debugging.
    """
    try:
        from langsmith import traceable
        load_dotenv()  # Load .env file for LangSmith API key
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
        os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "default_project")
        os.environ["LANGSMITH_RUN_NAME"] = os.getenv("LANGSMITH_RUN_NAME", "rag_pdf_chat")
        os.environ["LANGSMIT_TRACING"] = "true"
        return True
    except ImportError:
        return False


# --- Environment Setup ---
def load_environment():
    """
    Loads environment variables from .env file and sets necessary tokens.
    """
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


# --- Streamlit Page Setup ---
def setup_streamlit_page():
    """
    Configure Streamlit page layout and title.
    """
    st.set_page_config(layout="wide")
    st.title("Conversational RAG over PDFs")
    st.markdown("Upload PDFs and chat with their content using RAG and memory!")


# --- Retrieve or Input API Key ---
def get_groq_api_key():
    """
    Retrieves the GROQ API key from environment or asks user input.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your GROQ API key:", type="password")
    else:
        st.success("Using GROQ API key from environment variables")
    return api_key


# --- Manage Chat History State ---
def initialize_chat_state():
    """
    Initialize the chat session state storage if it doesn't exist.
    """
    if "store" not in st.session_state:
        st.session_state.store = {}


def get_session_history(session: str) -> BaseChatMessageHistory:
    """
    Returns the ChatMessageHistory for a given session id,
    creating a new one if it doesn't exist.
    """
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# --- PDF Processing ---
def process_pdfs(files):
    """
    Takes uploaded PDF files, saves them temporarily,
    loads and extracts documents & their text content.

    Args:
        files: List of uploaded files from Streamlit uploader.

    Returns:
        documents: List of LangChain Document objects for all PDFs.
        extracted_text: List of concatenated raw text extracted from each PDF.
    """
    documents = []
    extracted_text = []
    for i, uploaded_file in enumerate(files):
        temp_path = f"./temp_{i}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
            extracted_text.append("\n\n".join([doc.page_content for doc in docs]))
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
    return documents, extracted_text


# --- Display PDFs and Extracted Text in Sidebar ---
def display_pdf_and_text(uploaded_files, extracted_text):
    """
    Displays uploaded PDFs as embedded iframes along with
    a preview of their extracted text in the sidebar.

    Args:
        uploaded_files: List of uploaded PDF files.
        extracted_text: Corresponding list of extracted text strings.
    """
    with st.sidebar:
        st.markdown("### Uploaded PDFs and Extracted Text")
        for i, file in enumerate(uploaded_files):
            st.markdown(f"**{file.name}**")
            file.seek(0)  # Reset file pointer before reading for preview
            base64_pdf = base64.b64encode(file.read()).decode('utf-8')
            pdf_display = (
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                'width="100%" height="400" type="application/pdf"></iframe>'
            )
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.markdown("**Extracted Text Preview:**")
            st.text_area("Text", value=extracted_text[i][:1000], height=150)


# --- Build RAG Chain ---
def build_rag_chain(llm, documents):
    """
    Constructs the Retriever-Augmented Generation (RAG) chain with
    history-aware retriever and document question-answering.

    Args:
        llm: The language model instance.
        documents: List of LangChain Document objects.

    Returns:
        conversational_rag_chain: RunnableWithMessageHistory chain ready for conversation.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3.2")
    persist_path = "./chroma_store"
    if not os.path.exists(persist_path):
        os.mkdir(persist_path)
    vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory=persist_path)
    retriever = vectorstore.as_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference context in the chat history, "
         "formulate a standalone question. Do NOT answer it. Only reformulate."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the following pieces of retrieved context to answer "
         "the question. If unsure, say you don't know. Be concise (max 3 sentences).\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain


# --- Main Application Flow ---
def main():
    # Setup
    tracing_enabled = setup_langsmith_tracing()
    load_environment()
    setup_streamlit_page()

    # API key & session id inputs
    api_key = get_groq_api_key()
    session_id = st.text_input("Session ID", value="default_session")

    initialize_chat_state()

    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                documents, extracted_texts = process_pdfs(uploaded_files)

            display_pdf_and_text(uploaded_files, extracted_texts)

            if documents:
                conversational_rag_chain = build_rag_chain(llm, documents)

                user_question = st.text_input("Ask your question:")
                if user_question:
                    history = get_session_history(session_id)
                    with st.spinner("Thinking..."):
                        response = conversational_rag_chain.invoke(
                            {"input": user_question},
                            config={"configurable": {"session_id": session_id}}
                        )
                    st.success("Assistant:")
                    st.write(response["answer"])

                    with st.expander("View Chat History"):
                        for msg in history.messages:
                            st.markdown(f"**{msg.type.capitalize()}**: {msg.content}")

            else:
                st.warning("No valid documents were processed.")
    else:
        st.warning("Please enter your GROQ API key to proceed.")


if __name__ == "__main__":
    main()
