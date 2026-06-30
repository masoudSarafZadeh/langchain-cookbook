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
├── example.env          # Environment credential setup template
├── main.py               # Combined ingestion and dual-agent runtime script
├── README.md             # Implementation instructions
└── requirements.txt      # Module dependencies (Chroma, PyPDF, Gemini)
```
## Setup & Execution

### 1. Navigate and Install
```bash
cd RAG-agent
pip install -r requirements.txt
```
### 2. Prepare Environment Variables
Copy the template configuration:

```bash
cp example.env
```
Open your new .env file and supply your key:

```plaintext
GOOGLE_API_KEY=your_gemini_api_key_here
```
### 3. Add Your Document
Create a data/ folder if it doesn't exist.

Drop your target PDF file inside it.

Open main.py and update the PDF_FILE_PATH variable to point to your file (e.g., data/my_book.pdf).

### 4. Run the Script
```bash
python main.py
```
