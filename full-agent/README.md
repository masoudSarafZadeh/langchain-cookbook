# Hierarchical Deep Agent with Task Tracking

This sub-repository contains a fully modular, hierarchical multi-agent system built using modern LangChain and LangGraph. 

Unlike standard, flat ReAct agents that struggle with context pollution during long-running tasks, this project implements a **Deep Agent Architecture**. It uses a central supervisor agent to coordinate workflows, track progress via a built-in TODO list, and isolate heavy research tasks out to specialized sub-agents.

## Project Architecture

The agent's brain is broken down into a clean, separated module system:

* **`main.py`**: The entry point that orchestrates the initialization, builds the graph execution flows, and handles user queries.
* **`src/state.py`**: The single source of truth (`DeepAgentState`) managing conversational history, local virtual file frames, and the live task tracker.
* **`src/todo_tools.py` & `src/file_tools.py`**: The manager's tools for modifying project checklists and reading/writing files.
* **`src/research_tools.py`**: High-efficiency lookup capabilities (such as Tavily search) isolated specifically for research nodes.
* **`src/task_tool.py`**: The delegation bridge allowing the manager to spin up independent sub-agents dynamically.

## Directory Layout

```text
full_agent/
├── src/
│   ├── __init__.py        # Marks src as a importable Python package
│   ├── file_tools.py      # virtual file interaction logic
│   ├── prompts.py         # System prompts and tool instruction sets
│   ├── research_tools.py  # Web data retrieval pipelines
│   ├── state.py           # Core graph state definitions
│   ├── task_tool.py       # Sub-agent factory abstractions
│   └── todo_tools.py      # State-driven checklist modifiers
├── .env.example           # Environment template configuration
├── main.py                # Main orchestration application
├── README.md              # Project documentation
└── requirements.txt       # Local dependency pinning
