"""
LangChain RAG Example
Demonstrates two approaches to RAG: 
1. Using an Agent with a specific Retrieval Tool
2. Using Middleware to dynamically inject context into the prompt
"""

import os
import getpass
from dotenv import load_dotenv

# LangChain Imports
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# ==========================================
# 1. ENVIRONMENT & CONFIGURATION
# ==========================================
load_dotenv(os.path.join("..", ".env"), override=True)
os.environ["LANGSMITH_TRACING"] = "false"


if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

PDF_FILE_PATH = "pdf dir..."
CHROMA_PERSIST_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "example_collection"

# ==========================================
# 2. INITIALIZE MODELS & VECTOR STORE
# ==========================================
print("Initializing models and vector store...")

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR, 
)

# ==========================================
# 3. DOCUMENT INGESTION & SPLITTING
# ==========================================
print(f"Loading document from: {PDF_FILE_PATH}")
loader = PyPDFLoader(PDF_FILE_PATH)
docs = loader.load()

print(f"Loaded {len(docs)} pages. Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Adding {len(all_splits)} chunks to Chroma vector store...")
ids = vector_store.add_documents(documents=all_splits)

# Optional: Test similarity search directly
print("\n--- Testing Vector Store Search ---")
test_results = vector_store.similarity_search("توبه اجتماعی")
if test_results:
    print(f"Top result excerpt: {test_results[0].page_content[:150]}...")


# ==========================================
# 4. APPROACH 1: RAG VIA AGENT TOOL
# ==========================================
print("\n=== Running RAG Approach 1: Agent Tool ===")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tool_agent_prompt = (
    "You have access to a tool that retrieves context from a pdf of a book. "
    "Use the tool to help answer user queries. Answer in Persian."
)

tool_agent = create_agent(model, tools=[retrieve_context], system_prompt=tool_agent_prompt)
test_query = "تسویف چیست؟"

for event in tool_agent.stream(
    {"messages": [{"role": "user", "content": test_query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()


# ==========================================
# 5. APPROACH 2: RAG VIA MIDDLEWARE
# ==========================================
print("\n=== Running RAG Approach 2: Middleware Context Injection ===")

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context directly into the system prompt based on the last user query."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)
    
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    system_message = (
        "You are a helpful assistant. Use the following context in your response in Persian language:"
        f"\n\n{docs_content}"
    )
    return system_message

middleware_agent = create_agent(model, tools=[], middleware=[prompt_with_context])

for step in middleware_agent.stream(
    {"messages": [{"role": "user", "content": test_query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
