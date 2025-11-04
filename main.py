import os
import tempfile
import shutil
import uuid
import re  # <-- NEW IMPORT for text cleaning
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google.api_core import exceptions as google_exceptions
from typing import List

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma

# --- IMPORTS FOR RE-RANKER ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
# --- END IMPORTS ---


# --- Application Setup ---

# Check for Google Service Account credentials
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    print("INFO:     Found GOOGLE_APPLICATION_CREDENTIALS. Service account will be used for authentication.")
else:
    print("WARNING:  GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    print("WARNING:  Please set it to the path of your service account JSON file.")
    print("WARNING:  Example: export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/your/confidential.json\"")

# Check for Cohere API Key
if "COHERE_API_KEY" not in os.environ:
    print("WARNING:  COHERE_API_KEY environment variable not set.")
    print("WARNING:  The Re-ranking Retriever will fail. Get a key from cohere.ai")


# Temporary directory to store uploaded files
TEMP_DIR = tempfile.mkdtemp()

app = FastAPI(
    title="Chat with your PDF API",
    description="An API to upload a PDF and ask questions about its content using LangChain and Gemini.",
    version="3.0-text-clean-fix" # Version bump
)

# Global variables to hold the vector store and RAG chain
vector_store: Chroma | None = None
rag_chain = None
base_retriever = None # Global for debugging

# --- Pydantic Models for API ---

class AskRequest(BaseModel):
    """Request model for the /ask endpoint"""
    question: str

# --- Helper Functions ---

class SafeGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Wrapper around GoogleGenerativeAIEmbeddings to ensure BOTH
    'embed_query' and 'embed_documents' return plain list[float]
    to be compatible with ChromaDB.
    """
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text and ensure the output is a plain list.
        """
        embedding = super().embed_query(text)
        return list(embedding)

    # --- FIX 2: Added **kwargs to accept 'task_type' ---
    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Embed a list of documents and ensure the output is a plain list of lists.
        """
        # Pass all arguments (including the hidden 'task_type') to the parent
        embeddings = super().embed_documents(texts, **kwargs) 
        # Force-cast each embedding to a simple list
        return [list(e) for e in embeddings]


def load_and_split_pdf(file_path: str) -> list:
    """Loads a PDF, cleans its text, and splits it into chunks."""
    print(f"Loading and splitting document: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        print("Cleaning extracted text...")
        for page in pages:
            # This regex finds a character followed by a space,
            # and replaces it with just the character.
            # e.g., "H e l l o" -> "Hello"
            page.page_content = re.sub(r"(.) \b", r"\1", page.page_content)
            
            # --- ADD THIS LINE ---
            # Replace single newlines with a space to join broken words
            page.page_content = re.sub(r"\n", " ", page.page_content)
            # --- END OF NEW LINE ---

            # Also replace multiple spaces (which may have been created
            # by the step above) with a single space
            page.page_content = re.sub(r" +", " ", page.page_content)
        print("Text cleaning complete.")
        # --- END OF FIX 1 ---

        # Sticking with 500/100 as this is a good, specific chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(pages)
        print(f"Successfully split document into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error in load_and_split_pdf: {e}")
        raise

def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Creates an in-memory vector store from text chunks.
    This function now implements manual batching to respect API limits.
    """
    global base_retriever # We will set the retriever here
    print("Creating vector store...")
    try:
        # 1. Initialize our NEW, SAFE embeddings model
        embeddings_model = SafeGoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            max_retries=3  # Enable automatic retries
        )

        # 2. Initialize an EMPTY Chroma vector store first
        print("Initializing empty Chroma database...")
        db = Chroma(
            collection_name=f"pdf_chat_{uuid.uuid4()}", # Unique name for in-memory
            embedding_function=embeddings_model 
        )

        # 3. Manually add documents in batches of 100
        BATCH_SIZE = 100 
        print(f"Adding {len(chunks)} chunks to database in batches of {BATCH_SIZE}...")
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            print(f"  Adding batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1}...")
            
            # This call will now pass 'task_type' to our safe function
            db.add_documents(batch_chunks) 
        
        print("Vector store created and populated successfully.")
        
        # 4. Create the base retriever
        base_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 50}   # Get 50 candidates for the re-ranker
        )

        return db
    except Exception as e:
        print(f"Error in create_vector_store: {e}")
        raise

def create_rag_chain(db: Chroma):
    """Creates a RAG (Retrieval-Augmented Generation) chain."""
    global base_retriever 
    print("Creating RAG chain...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro-latest", 
            temperature=0,
            max_retries=3  # Also add retries to the LLM
        )

        # --- FIX 3: Removed the stray "MODIFIED" word ---
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise and based *only* on the provided context."
            "\n\n"
            "<context>"
            "{context}"
            "</context>"
        )
        # --- END OF FIX 3 ---

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        if base_retriever is None:
            print("ERROR: base_retriever is not set. Creating a default one.")
            base_retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 50}
            )

        # Initialize the Cohere Re-ranker
        cohere_reranker = CohereRerank()

        # Create the Contextual Compression Retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=cohere_reranker, 
            base_retriever=base_retriever
        )
        
        final_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
        
        print("RAG chain created successfully.")
        return final_chain
    except Exception as e:
        print(f"Error in create_rag_chain: {e}")
        raise

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint to check if the server is running."""
    return {
        "message": "Welcome to the Chat with your PDF API!",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, processes it, and creates the RAG chain."""
    global vector_store, rag_chain

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    file_path = ""
    try:
        safe_filename = file.filename.replace("..", "_").replace("/", "_")
        file_path = os.path.join(TEMP_DIR, safe_filename)
        
        print(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        
        file_size = os.path.getsize(file_path)
        print(f"File saved successfully. Size: {file_size} bytes.")
        if file_size == 0:
            print("WARNING: The uploaded file is empty.")
            raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

        chunks = load_and_split_pdf(file_path)
        vector_store = create_vector_store(chunks) # This now also sets the 'base_retriever'
        rag_chain = create_rag_chain(vector_store)
        
        return {"message": f"Successfully processed '{file.filename}'. Ready to answer questions."}

    except google_exceptions.ResourceExhausted as e:
        print(f"Google API quota error: {e}")
        raise HTTPException(status_code=400, detail=f"Google API quota exceeded. Please check your plan. Error: {e.message}")
    except Exception as e:
        print(f"Error during upload: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")

@app.post("/ask")
async def ask_question(request: AskRequest):
    """Asks a question to the processed document."""
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No document has been processed. Please upload a PDF to /upload first.")

    try:
        # Use the RAG chain to get an answer
        response = rag_chain.invoke({"input": request.question})
        
        # --- NEW DEBUGGING LOG ---
        print("\n--- COHERE RE-RANKER DEBUG ---")
        print(f"Question: {request.question}")
        retrieved_context = response.get("context", [])
        print(f"Context chunks found after re-ranking: {len(retrieved_context)}")
        for i, chunk in enumerate(retrieved_context):
            print(f"  CHUNK {i+1} (Source Page: {chunk.metadata.get('page', 'N/A')}):")
            print(f"  > {chunk.page_content[:250]}...") 
        print("--- END DEBUG ---\n")
        # --- END OF DEBUGGING LOG ---

        # The response is a dictionary, we just want the 'answer'
        return {"answer": response.get("answer", "No answer found.")}

    except google_exceptions.ResourceExhausted as e:
        print(f"Google API quota error during ask: {e}")
        raise HTTPException(status_code=400, detail=f"Google API quota exceeded. Please check your plan. Error: {e.message}")
    except Exception as e:
        print(f"Error during ask: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- NEW DEBUGGING ENDPOINT ---
@app.post("/debug_retriever")
async def debug_retriever(request: AskRequest):
    """
    A debug endpoint to see the raw chunks from the 'dumb' retriever
    before they go to the re-ranker.
    """
    if base_retriever is None:
        raise HTTPException(status_code=400, detail="No document has been processed. Please upload a PDF to /upload first.")
    
    try:
        # Get the Top 50 "dumb" results
        retrieved_chunks = await base_retriever.ainvoke(request.question)
        
        # Return just the text content for us to read
        chunk_texts = [chunk.page_content for chunk in retrieved_chunks]
        return {
            "question": request.question,
            "retrieved_chunk_count": len(chunk_texts),
            "chunks": chunk_texts
        }
    except Exception as e:
        print(f"Error during debug_retriever: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


