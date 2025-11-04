# Chat with your PDF

This project is a high-end portfolio piece demonstrating a complete, API-driven **Retrieval-Augmented Generation (RAG)** pipeline.

It solves one of the most common and valuable problems in the AI industry: how to "chat" with your own documents. The API allows a user to upload a PDF document and then ask questions about its content, receiving answers that are grounded *only* in the text of that document.

This is the core technology behind the advanced "AI Document Platform" gigs you see on freelance sites.



[  *(This is where you can add your video demo once you build the UI!)*]: #



## 🏛️ Architecture

This project is a simple, two-step API built with **FastAPI**:

1.  **`POST /upload`**: The user uploads a PDF file. The server:
    * Loads the PDF and splits it into small, overlapping text chunks using **LangChain**.
    * Embeds these chunks into vectors (numerical representations) using **OpenAI's** embedding model.
    * Saves these vectors into an in-memory **ChromaDB** vector store.
    * Creates a **RAG retrieval chain** and stores it in a global variable, ready for questions.

2.  **`POST /ask`**: The user sends a JSON request with a question (e.g., `{"question": "What is..."}`). The server:
    * Takes the question and uses the RAG chain's **retriever** to find the most relevant text chunks from the vector store.
    * "Stuffs" these chunks (the `context`) and the user's `input` (the question) into a specialized prompt.
    * Sends the final prompt to an **LLM (gpt-4o)** to generate an answer based *only* on the provided context.
    * Returns the final answer to the user as JSON.

## 🛠️ Tech Stack

* **Backend API:** **FastAPI**
* **Orchestration:** **LangChain**
* **LLM & Embeddings:** **OpenAI** (gpt-4o)
* **Vector Store:** **ChromaDB** (in-memory)
* **PDF Loading:** `pypdf` (via LangChain)
* **Server:** `uvicorn`

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone [https://github.com/Kalana-Gayan/pdf_chat_api.git](https://github.com/Kalana-Gayan/pdf_chat_api.git)
cd pdf_chat_api
```


### 2. Install Dependencies

Make sure you have Python 3.10+ installed. Then, install the required libraries from the requirements.txt file.
```bash
pip install -r requirements.txt
```

### 3. Set Your OpenAI and Cohere API Key

This project requires an OpenAI API key. You must set it as an environment variable in your terminal.

On macOS/Linux:
```Bash

export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="paste_your_key_here"
```
On Windows:
```Bash

set OPENAI_API_KEY="paste_your_key_here"
set COHERE_API_KEY="paste_your_key_here"
```
### 4. Run the API Server

Use uvicorn to run the FastAPI server.
``` Bash
uvicorn main:app --reload
```
The server will be running at http://127.0.0.1:8000. You can see the automatic API documentation at http://127.0.0.1:8000/docs.

## ⚙️ How to Use (API Endpoints)

You must upload a document before you can ask questions.

### 1. Upload a PDF

First, send a POST request to the /upload endpoint with your PDF file.

Using curl (from your terminal):
```Bash

curl -X POST "[http://127.0.0.1:8000/upload](http://127.0.0.1:8000/upload)" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```
Response (if successful):
```JSON

{
  "message": "Successfully uploaded and processed document.pdf."
}
```

### 2. Ask a Question

Now you can send POST requests to the /ask endpoint with your questions.

Using curl (from your terminal):
```Bash

curl -X POST "[http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"What does this document say about tinnitus?\"}"
```
Response (if successful):
```JSON
{
  "answer": "The document states that tinnitus is a persistent ringing in the ears, often associated with noise-induced hearing loss."
}
```
## 🌟 Next Steps

This is the core backend API. The next logical steps for this portfolio project are:

- Build a Web UI: Create a separate ui.py script using Gradio or Streamlit that provides a simple web-based chat interface that calls this API.
- Add Persistence: Modify the create_vector_store function to persist the ChromaDB to disk. This way, you don't have to re-upload the same document every time the server restarts.
