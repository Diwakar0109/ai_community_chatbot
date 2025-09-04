import os
import re
from dotenv import load_dotenv
from typing import List, Tuple, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()

# --- App Initialization ---
app = FastAPI(
    title="AI Chatbot Backend",
    description="Backend for a RAG chatbot with a strict, fixed knowledge base.",
    version="2.3.0" # Version updated for strict mode
)

# --- CORS CONFIGURATION ---
allowed_origins = [
    "https://ai-community-gray.vercel.app/", # Your production frontend

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables & Constants ---
COHERE_EMBED_MODEL = "embed-english-light-v3.0"
embeddings = None
inbuilt_vector_store = None # Will hold the single, global knowledge base
sessions: Dict[str, Dict] = {}


@app.on_event("startup")
async def startup_event():
    """
    Initializes the Cohere embeddings model and loads the inbuilt knowledge base.
    """
    print("--- Application Startup ---")
    global embeddings, inbuilt_vector_store
    
    # 1. Initialize Cohere embeddings
    print(f"Initializing Cohere embeddings model ('{COHERE_EMBED_MODEL}')...")
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("FATAL: COHERE_API_KEY environment variable not set.")
        
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model=COHERE_EMBED_MODEL)
        _ = embeddings.embed_query("Test query")
        print(f"✅ Cohere embeddings model initialized successfully.")
    except Exception as e:
        raise RuntimeError("Application startup failed: Embeddings could not be loaded.") from e

    # 2. Load and process the inbuilt memory file
    print("Loading inbuilt memory from 'inbuilt_memory.txt'...")
    try:
        inbuilt_memory_path = os.path.join(os.path.dirname(__file__), 'inbuilt_memory.txt')
        
        if not os.path.exists(inbuilt_memory_path):
            print(f"⚠️ WARNING: Knowledge base file not found at '{inbuilt_memory_path}'. Bot will not be able to answer questions.")
            return

        with open(inbuilt_memory_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if text.strip():
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            inbuilt_vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
            print("✅ Inbuilt memory loaded and vectorized successfully.")
        else:
            print("⚠️ WARNING: 'inbuilt_memory.txt' is empty. Bot will not have knowledge to answer questions.")

    except Exception as e:
        print(f"❌ ERROR: Could not load inbuilt memory: {e}")
        
    print("--- Application Ready ---")


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

class SessionRequest(BaseModel):
    session_id: str


# --- Helper Function ---
def create_rag_chain(vector_store):
    """Creates a ConversationalRetrievalChain with a very strict prompt."""
    llm = ChatGroq(temperature=0.1, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

    # THIS IS THE MOST IMPORTANT CHANGE. This prompt forbids external knowledge.
    prompt_template = """
    You are a specialized AI assistant. Your ONLY job is to answer questions based STRICTLY on the context provided below.

    Follow these rules without exception:
    1. Analyze the user's question.
    2. If the provided 'Context' contains the information to answer the question, formulate a clear answer using ONLY that information.
    3. If the 'Context' does NOT contain the answer, you MUST respond with the exact phrase: "The answer to your question is not in my knowledge base."
    4. DO NOT use any external knowledge. DO NOT make up information. Your knowledge is 100% limited to the text in the 'Context'.
    
    Context: {context}
    
    Question: {question}
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

# --- API Endpoints ---
@app.post("/session/clear", summary="Clear a session's chat history")
async def clear_session(request: SessionRequest):
    if request.session_id in sessions:
        sessions.pop(request.session_id, None)
        print(f"Cleared chat history for session: {request.session_id}")
    return {"status": "success", "message": "Session chat history cleared."}


@app.post("/chat", response_model=ChatResponse, summary="Handle a chat message for a specific session")
async def chat_with_bot(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is missing.")

    # --- Step 1: Handle simple greetings separately ---
    if re.search(r"^\s*(hi|hello|hey|good morning|good afternoon)\s*$", request.question, re.IGNORECASE):
        return ChatResponse(answer="Hello! How can I help you with the information I have?")

    # --- Step 2: Handle meta-questions ---
    history_question_pattern = r"(what was|what's|what is)\s+(my|the)\s+(last|previous)\s+question|my\s+previous\s+question"
    if re.search(history_question_pattern, request.question, re.IGNORECASE):
        if request.chat_history:
            last_user_question = request.chat_history[-1][0]
            return ChatResponse(answer=f"Your previous question was: \"{last_user_question}\"")
        else:
            return ChatResponse(answer="You haven't asked any questions before this one.")

    # --- Step 3: Process all other questions through the strict RAG chain ---
    try:
        # Check if the knowledge base was loaded successfully on startup
        if not inbuilt_vector_store:
            return ChatResponse(answer="I'm sorry, my knowledge base is not available. Please contact the administrator.")

        # Create a RAG chain for the session if it's the first message
        if request.session_id not in sessions:
            sessions[request.session_id] = {
                "rag_chain": create_rag_chain(inbuilt_vector_store)
            }
        
        # Use the RAG chain to get a response based ONLY on the inbuilt text file
        session = sessions[request.session_id]
        result = session["rag_chain"].invoke({
            "question": request.question,
            "chat_history": request.chat_history
        })
        return ChatResponse(answer=result["answer"])

    except Exception as e:
        print(f"Error during chat for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    # reload=False is recommended when using startup events to avoid re-running them
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
