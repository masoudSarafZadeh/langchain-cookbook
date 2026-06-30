# Dual-Approach Retrieval-Augmented Generation (RAG)

This sub-repository provides a hands-on cookbook example demonstrating two distinct patterns for implementing Retrieval-Augmented Generation (RAG) using modern LangChain and Google Gemini models.

The script ingests a PDF document, splits its text content into semantic chunks, stores it inside an on-disk Chroma vector database, and queries it in Persian.

## Architectures Compared

This recipe implements and showcases two different RAG methods:

1. **Approach 1: Tool-Driven RAG (Agent Choice)**
   * The retriever is exposed to the model as an active `@tool`. 
   * The LLM autonomously decides *when* to invoke the tool, *what* search query to write, and *how* to incorporate the returned metadata into its thought stream.
   * **Best for:** Complex workflows where data lookup is conditional.

2. **Approach 2: Middleware RAG (Context Injection)**
   * Uses LangChain's `@dynamic_prompt` middleware hook.
   * Every time a user sends a message, the system automatically intercepts the query behind the scenes, runs a similarity search, and injects the raw context into the system prompt before the model even reads it.
   * **Best for:** Standard, predictable Q&A systems where lookup is required on every single turn.

## Project Structure

```text
RAG-agent/
├── .env.example          # Environment credential setup template
├── main.py               # Combined ingestion and dual-agent runtime script
├── README.md             # Implementation instructions
└── requirements.txt      # Module dependencies (Chroma, PyPDF, Gemini)
