import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# IMPORTANT CHANGE: Using GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# IMPORTANT CHANGE: Using ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import glob
import nest_asyncio
nest_asyncio.apply()

# --- Configuration and Environment Variables ---
# --- DIRECTORY FOR PDF FILES ---
PDF_DIRECTORY = "CP_Toolkit/" # <--- CHANGE THIS TO YOUR PDF DIRECTORY PATH

if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY environment variable not set. Please set it before running the app.")
    st.stop() # Stop the app if the key is not set

GOOGLE_GEMINI_MODEL = "gemini-2.5-flash-lite" # Using gemini-flash for efficiency

# --- Functions for RAG Pipeline ---

def load_pdfs_from_directory(directory_path):
    """
    Loads documents from all PDF files within a specified directory.
    """
    all_documents = []
    if not os.path.exists(directory_path):
        st.error(f"Error: Directory '{directory_path}' does not exist.")
        return []

    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    if not pdf_files:
        st.warning(f"No PDF files found in directory: '{directory_path}'")
        return []

    for pdf_file_path in pdf_files:
        st.info(f"Loading {os.path.basename(pdf_file_path)}...")
        try:
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            st.error(f"Could not load PDF '{os.path.basename(pdf_file_path)}': {e}")
    return all_documents

def split_documents(documents):
    """
    Splits documents into smaller, manageable chunks.
    This is crucial for fitting content into the LLM's context window
    and for effective retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Maximum size of each text chunk
        chunk_overlap=200,    # Overlap between chunks to maintain context
        length_function=len,  # Function to calculate chunk length
        add_start_index=True, # Add metadata about the starting character index
    )
    st.info("Splitting documents into chunks...")
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    """
    Creates a FAISS vector store from document chunks using GoogleGenerativeAIEmbeddings.
    """
    st.info(f"Creating embeddings using Google Gemini Embeddings and building vector store...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001") # Recommended embedding model for Gemini
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create embeddings with Google Gemini. Check your API key and network connection. Error: {e}")
        return None

def initialize_llm():
    """
    Initializes the Google Gemini Flash LLM.
    """
    st.info(f"Initializing Google Gemini Flash LLM with model '{GOOGLE_GEMINI_MODEL}'...")
    try:
        llm = ChatGoogleGenerativeAI(model=GOOGLE_GEMINI_MODEL, temperature=0.5)
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Google Gemini Flash LLM. Check your API key and network connection. Error: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="CPRN Chatbot using Gemini Flash", layout="wide")
st.title("📄 CPRN Chatbot")
st.markdown(
    f"This chatbot is configured to read PDF documents from the directory: `{PDF_DIRECTORY}`. "
    f"It uses Langchain for RAG and leverages **Google Gemini Flash** ({GOOGLE_GEMINI_MODEL} model) for both embeddings and the Large Language Model. "
    
)

# Initialize session state variables to persist data across reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_directory" not in st.session_state:
    st.session_state.processed_directory = None

# Button to process documents from the predefined directory
if st.button("Process Documents from Directory"):
    # Only re-process if the directory path has changed or if not processed yet
    if st.session_state.processed_directory != PDF_DIRECTORY or not st.session_state.qa_chain:
        with st.spinner(f"Processing documents from '{PDF_DIRECTORY}'... This might take a moment."):
            try:
                # Load documents from the specified directory
                documents = load_pdfs_from_directory(PDF_DIRECTORY)

                if not documents:
                    st.warning("No documents were loaded. Please check the directory path and ensure it contains PDFs.")
                    st.session_state.qa_chain = None # Reset QA chain if no documents
                    st.session_state.processed_directory = None
                else:
                    st.info(f"Successfully loaded {len(documents)} pages from PDFs in '{PDF_DIRECTORY}'.")

                    # Split documents into chunks
                    chunks = split_documents(documents)
                    st.info(f"Documents split into {len(chunks)} text chunks.")

                    # Create vector store from chunks
                    st.session_state.vector_store = create_vector_store(chunks)
                    if st.session_state.vector_store:
                        st.success("Vector store created successfully! Ready for retrieval.")
                    else:
                        st.error("Vector store creation failed. Please check your Google API Key and network.")
                        st.session_state.qa_chain = None
                        st.session_state.processed_directory = None

                    # Initialize LLM
                    if st.session_state.vector_store: # Only initialize LLM if vector store was successfully created
                        st.session_state.llm = initialize_llm()
                        if st.session_state.llm:
                            st.success("Large Language Model (LLM) initialized.")
                            # Set up RetrievalQA chain
                            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                                llm=st.session_state.llm,
                                chain_type="stuff", # "stuff" combines all retrieved documents into one prompt
                                retriever=st.session_state.vector_store.as_retriever(),
                                return_source_documents=True # Important for showing where the answer came from
                            )
                            st.success("RAG system is ready! You can now ask questions about your documents.")
                            st.session_state.processed_directory = PDF_DIRECTORY # Store the processed directory
                        else:
                            st.error("LLM initialization failed. Please check your Google API Key and network.")
                            st.session_state.qa_chain = None # Reset QA chain if LLM fails
                    else:
                        st.error("Cannot initialize LLM as vector store creation failed.")
                        st.session_state.qa_chain = None


            except Exception as e:
                st.error(f"An error occurred during document processing: {e}")
                # Reset session state on error to allow reprocessing
                st.session_state.vector_store = None
                st.session_state.llm = None
                st.session_state.qa_chain = None
                st.session_state.processed_directory = None # Clear processed directory on error
    else:
        st.info("Documents already processed from this directory. Ready to chat!")


# Section for asking questions
if st.session_state.qa_chain:
    st.markdown("---")
    st.header("Ask a Question")
    user_query = st.text_input(
        "Enter your question here:",
        placeholder="e.g., What are the main features of the product?",
        key="user_query_input" # Unique key for the text input
    )

    if user_query:
        with st.spinner("Generating response..."):
            try:
                # Invoke the RAG chain with the user's query
                response = st.session_state.qa_chain.invoke({"query": user_query})

                st.subheader("Answer:")
                st.write(response["result"]) # Display the generated answer

                # Display source documents if available
                if response.get("source_documents"):
                    st.subheader("Sources:")
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Document {i+1}:**")
                        # Display metadata (page number, source file)
                        st.write(f"Page: {doc.metadata.get('page', 'N/A')}")
                        st.write(f"Source File: {os.path.basename(doc.metadata.get('source', 'N/A'))}") # Show only file name
                        # Display a snippet of the relevant content
                        st.code(doc.page_content[:500] + "...", language="text")
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
else:
    st.info("Click 'Process Documents from Directory' to load the data and enable the chatbot.")
