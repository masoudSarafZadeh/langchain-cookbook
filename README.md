# LangChain & LangGraph Agent Cookbook

Welcome to the ultimate repository of modular, production-grade AI agent architectures. This cookbook contains practical, standalone recipes demonstrating advanced cognitive design patterns using **LangChain** and **LangGraph**.

Rather than simple, toy examples, each sub-repository houses a self-contained, real-world agent pipeline with structured state handling, resilient fallbacks, and multi-model integrations.

---

## Repository Architecture

This ecosystem is split into three highly specialized agent blueprints. You can navigate into any sub-directory to find dedicated setup manuals and visual workflow graphs:

```text
langchain-cookbook/
├── full_agent/          # Hierarchical Deep Agent with Task Tracking
├── RAG-agent/           # Dual-Approach RAG: Tool-driven vs Prompt Middleware
├── SQL-agent/           # Autonomous Text-to-SQL with Self-Correction
├── LICENSE              # Repository open-source licensing terms
└── README.md            # Global workspace registry (This file)

##  The Recipes At A Glance

### 1. Hierarchical Deep Agent
* **Core Stack:** LangChain, LangGraph
* **Design Pattern:** Supervisor-Worker Hierarchy
* **Key Feature:** Solves token bloat and context pollution by utilizing an autonomous supervisor agent that dynamically coordinates tasks, tracks progress with a live digital TODO checklist, and spins up isolated worker frames for background research.

### 2. Dual-Approach RAG Agent
* **Core Stack:** ChromaDB, PyPDF
* **Design Pattern:** Tool Ingestion vs Prompt Middleware Injection
* **Key Feature:** Directly compares two distinct RAG architectures. One allows the LLM to choose when to deploy a search tool, while the other intercepts user input via custom `@dynamic_prompt` middleware to seamlessly inject PDF vector context before inference.

### 3. Autonomous SQL Agent
* **Core Stack:** LangGraph, PostgreSQL
* **Design Pattern:** Self-Correction & Fuzzy Fallback Loops
* **Key Feature:** A robust Text-to-SQL sales assistant. Features an automated syntax-checking engine that fixes raw queries before execution, grades retrieved results, and leverages regex fuzzy fallbacks (`LIKE`) to ensure the end application interface never gets blank data screens.

---

##  Global Getting Started

Every recipe in this cookbook is completely modular and self-contained. To run any agent locally:

1. **Clone this repository** to your machine.
2. **Open your terminal** and change into the specific recipe directory:
   ```bash
   cd SQL-agent # Or full_agent / RAG-agent
