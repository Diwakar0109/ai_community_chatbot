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
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration Constants ---
KNOWLEDGE_BASE_FILE = "inbuilt_memory.txt"
EMBED_MODEL = "embed-english-light-v3.0"
LLM_MODEL = "llama-3.1-8b-instant"
RETRIEVER_K = 4

# --- App Initialization ---
app = FastAPI(
    title="AI Community Chatbot",
    description="A concise and intelligent RAG chatbot for the AI Community at Bannari Amman Institute of Technology.",
    version="4.1.0" # Concise & Polished version
)
# (CORS configuration remains the same)
allowed_origins = [
    "https://bit-ai-community.vercel.app",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Global Variables ---
embeddings, vector_store, sessions = None, None, {}

# --- Helper Function for Structure-Aware Chunking ---
def split_text_by_project(text: str) -> List[str]:
    chunks = re.split(r'(?m)(?=^\d\s*-\s*)', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# --- Application Lifecycle Events (remains the same) ---
@app.on_event("startup")
async def startup_event():
    # This function is unchanged and correct.
    print("--- Application Startup ---")
    global embeddings, vector_store
    print(f"Initializing Cohere embeddings model ('{EMBED_MODEL}')...")
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY"); embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model=EMBED_MODEL); print("✅ Cohere embeddings model initialized successfully.")
    except Exception as e: raise RuntimeError(f"Application startup failed: Embeddings could not be loaded. Error: {e}")
    print(f"Loading knowledge base from '{KNOWLEDGE_BASE_FILE}'...")
    try:
        inbuilt_memory_path = os.path.join(os.path.dirname(__file__), KNOWLEDGE_BASE_FILE)
        if not os.path.exists(inbuilt_memory_path): return print(f"⚠️ WARNING: Knowledge base file not found at '{inbuilt_memory_path}'.")
        with open(inbuilt_memory_path, "r", encoding="utf-8") as f: text = f.read()
        if text.strip(): chunks = split_text_by_project(text); print(f"✅ Successfully split text into {len(chunks)} project-based chunks."); vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings); print("✅ Knowledge base loaded and vectorized successfully.")
        else: print(f"⚠️ WARNING: '{KNOWLEDGE_BASE_FILE}' is empty.")
    except Exception as e: print(f"❌ ERROR: Could not load knowledge base: {e}")
    print("--- Application Ready ---")

# --- Pydantic Models for API (remains the same) ---
class ChatRequest(BaseModel):
    session_id: str; question: str; chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

# --- Core RAG Chain Creation ---
def create_rag_chain(vectorstore):
    llm = ChatGroq(temperature=0.1, model_name=LLM_MODEL, groq_api_key=os.getenv("GROQ_API_KEY"))

    _template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Focus on the 'Follow Up Input'. If the user asks about "you" or "your purpose", the standalone question should be about the AI assistant itself.
    Otherwise, combine the follow-up with the context from the chat history.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # ==================== NEW, GENERALIZED FORMATTING PROMPT ====================
    main_prompt_template = """
    You are a precise and helpful AI assistant for the AI Community at Bannari Amman Institute of Technology.
    Your knowledge is STRICTLY limited to the information in the 'Context'. Your primary goal is to provide clear, structured, and easy-to-read answers.

    **Your Core Rules:**
    1.  **Be Brief and Direct:** Answer ONLY what the user asks. Provide concise, to-the-point answers.
    2.  **No External Knowledge:** NEVER use any information outside of the provided 'Context'.
    3.  **Handle Unknowns:** If the answer is not in the context, reply with: "I'm sorry, that information is not available in my knowledge base."

    **Formatting Excellence (MANDATORY):**
    *   **Default to Markdown Lists:** For ANY answer that involves enumerating two or more items (e.g., project names, features, objectives, benefits, locations), YOU MUST use bullet points (`*`). This makes information scannable and user-friendly.
    *   **Use Bold for Emphasis:** Always use **bold markdown** for project titles and other key terms to make them stand out.
    *   **Avoid Introductory Fluff:** Get straight to the point. Do not add conversational filler like "Here are the projects..." or "Certainly, the objectives are...".

    Context:
    {context}

    Question: {question}

    Structured and Helpful Answer:"""
    # =======================================================================
    
    MAIN_PROMPT = PromptTemplate(template=main_prompt_template, input_variables=["context", "question"])
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": MAIN_PROMPT},
    )

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse, summary="Handle a chat message")
async def chat_with_bot(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
    
    question_lower = request.question.lower().strip()
    
    # Expanded patterns for more robust conversational handling
    greeting_pattern = r"^\s*(hi|hii|hello|hey|heyy)\s*!*$"
    identity_pattern = r"who are you|what are you|what is your purpose|tell me about yourself"
    
    # --- NEW: Rule to handle farewells ---
    farewell_pattern = r"bye|goodbye|good bye|thanks|thank you|see you|ok then good bye"

    if re.fullmatch(greeting_pattern, question_lower):
        return ChatResponse(answer="Hello! I'm the AI assistant for the AI Community at Bannari Amman Institute of Technology. How can I help you?")

    if re.search(identity_pattern, question_lower):
        identity = (
            "I am an AI assistant for the **AI Community at Bannari Amman Institute of Technology**. "
            "My purpose is to answer questions about the community's projects based on my knowledge base."
        )
        return ChatResponse(answer=identity)

    # Add the new farewell check here
    if re.search(farewell_pattern, question_lower):
        farewell_message = "You're welcome! Goodbye and have a great day."
        return ChatResponse(answer=farewell_message)
    # --- End of new rule ---
    
    try:
        if not vector_store:
            return ChatResponse(answer="I'm sorry, my knowledge base is currently unavailable.")
            
        if request.session_id not in sessions:
            print(f"Creating new RAG chain for session: {request.session_id}")
            sessions[request.session_id] = {"rag_chain": create_rag_chain(vector_store)}
        
        rag_chain = sessions[request.session_id]["rag_chain"]
        result = rag_chain.invoke({
            "question": request.question,
            "chat_history": request.chat_history
        })
        
        return ChatResponse(answer=result["answer"].strip())
        
    except Exception as e:
        print(f"Error during chat for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
# (Main entry point remains the same)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"--- Starting server on http://0.0.0.0:{port} ---")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
