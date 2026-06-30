import json
import os
import re
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. ENVIRONMENT & CONFIGURATION
# ==========================================
load_dotenv(os.path.join("..", ".env"), override=True)
os.environ["LANGSMITH_TRACING"] = "false"

# Initialize OpenRouter Model
model = ChatOpenRouter(model="openai/gpt-oss-120b:free")
grader_model = ChatOpenRouter(model="openai/gpt-oss-120b:free")

# Load Database securely from Environment Variables
DATABASE_URL = os.getenv("DATABASE_URL")
db = SQLDatabase.from_uri(DATABASE_URL, include_tables=["table_name"])

# Initialize Toolkit and Extraction Rules
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")
run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

# ==========================================
# 2. STATE & GRAPH DEFINITIONS
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_db_data: Any  # Stores uncleaned DB results (including images)

class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

# ==========================================
# 3. HELPER FUNCTIONS & NODE LOGIC
# ==========================================
def run_and_parse(sql_query: str):
    try:
        res = db.run(sql_query)
        if isinstance(res, str):
            try:
                return json.loads(res)
            except json.JSONDecodeError:
                return eval(res)
        return res
    except Exception as e:
        print(f"Database Error: {e}")
        return []

def custom_run_query_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    call_id = tool_call["id"]
            
    data = run_and_parse(query)
    
    # Fallback Mechanism: If no products were found on fuzzy queries
    if not data or (isinstance(data, list) and len(data) == 0):
        if "recommended" in query or "best quality" in query:
            print("⚠️ No items matched specialized tags. Falling back to fuzzy keyword match...")
            product_keyword = "%"
            match = re.search(r"LIKE\s+['\"](.*?)['\"]", query, re.IGNORECASE)
            if match:
                product_keyword = match.group(1)
            
            fallback_query = f"SELECT * FROM goods WHERE product_name LIKE '{product_keyword}' LIMIT 5"
            data = run_and_parse(fallback_query)

    # Clean the payload context going to the LLM context to preserve tokens
    clean_data_for_llm = []
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                clean_row = {k: v for k, v in row.items() if k not in ["image"]}
                clean_data_for_llm.append(clean_row)
            else:
                clean_data_for_llm.append(row)
    else:
        clean_data_for_llm = data

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(clean_data_for_llm, ensure_ascii=False), 
                name="sql_db_query", 
                tool_call_id=call_id
            )
        ],
        "raw_db_data": data  # Keeps safe copy with image URLs for end application
    }

def list_tables(state: AgentState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}

def call_get_schema(state: AgentState):
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ==========================================
# 4. PROMPTS & COGNITIVE WORKFLOWS
# ==========================================
generate_query_system_prompt = """
You are an AI agent designed to translate natural-language requests into syntactically correct {dialect} SELECT queries for a grocery application.
Your primary role is to call the appropriate query execution tool with your generated SQL query.

CRITICAL SCHEMA RULES:
- Always use the exact snake_case columns as defined in the table schema. Never replace underscores with spaces (e.g., use price_after_off, NEVER "price after off").
- You MUST always use `SELECT *` from the "goods" table to ensure all product metadata and image URLs are fully retrieved.
- Limit results to at most {top_k} rows unless the user explicitly requests a different number. Always append a `LIMIT` clause.
- Use double quotes `""` for column identifiers if required, and single quotes `''` for string literals (e.g., product_name LIKE '%رب%').

Business Logic & Filtering Rules:
1. **Cheapest items**: If the user asks for the "cheapest" or "lowest price", filter using `product_name LIKE '%X%'` and order by the numeric column `price_after_off` in ASCENDING order (`ORDER BY price_after_off ASC`).
2. **Best Quality**: If the user asks for "best quality" or "highest quality", filter rows where the `llm_guide` column equals 'best quality'.
3. **Recommended items**: If the user asks for "recommended" items, or asks for a product generally without specifying a brand/weight, filter rows where the `llm_guide` column equals 'recommended'.
4. **Farsi Search**: When users search for an item in Persian (e.g., "رب گوجه"), write the raw string filter explicitly in the query using wildcards, for example: `WHERE product_name LIKE '%رب گوجه%'`. Do not use parameterized placeholders like `:q`.

Example Tool Arguments:
- User: "Give me the 5 cheapest spaghetti"
  ➔ Tool Query argument: SELECT * FROM goods WHERE product_name LIKE '%spaghetti%' ORDER BY price_after_off ASC LIMIT 5;
""".format(dialect=db.dialect, top_k=5)

def generate_query(state: AgentState):
    system_message = {"role": "system", "content": generate_query_system_prompt}
    llm_with_tools = model.bind_tools([run_query_tool])
    messages_to_send = [system_message] + state["messages"]    
    response = llm_with_tools.invoke(messages_to_send)
    return {"messages": [response]}
    
check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes. Rewrites must maintain tool call logic.
You MUST execute the final verified query by making a valid tool call to the query execution tool. Pass the clean SQL string as the tool argument.
""".format(dialect=db.dialect)

def check_query(state: AgentState):
    system_message = {"role": "system", "content": check_query_system_prompt}
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}

GRADE_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question.
Here is the retrieved document: 
{context} 
Here is the user question: {question} 
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
"""

def grade_documents(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(  
        [{"role": "user", "content": prompt}]
    )
    return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

REWRITE_PROMPT = """
You are an expert query optimizer for an online grocery shop. Strip conversational filler and rewrite for optimized database searching.
Initial user question: {question}
"""

def rewrite_question(state: AgentState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

GENERATE_PROMPT = """
You are a brilliant, friendly, and honest sales assistant for a grocery mobile application. 
Write a short, engaging response in Persian based strictly on the provided database context.
Question: {question}
Query Executed: {query}
Database Context: {context}
Response (in Persian):
"""

def generate_answer(state: AgentState):
    db_rows_content = None
    query = "Unknown query"
    messages = state["messages"]
    
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage) and msg.name == "sql_db_query":
            db_rows_content = msg.content
            if i > 0 and hasattr(messages[i-1], 'tool_calls') and messages[i-1].tool_calls:
                query = messages[i-1].tool_calls[0]["args"].get("query", "Unknown query")
            break

    if not db_rows_content:
        return {"messages": [AIMessage(content="هیچ داده‌ای در پایگاه داده یافت نشد.")]}

    question = state["messages"][0].content
    prompt = GENERATE_PROMPT.format(question=question, query=query, context=db_rows_content)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [AIMessage(content=response.content)]}

def should_continue(state: AgentState) -> Literal[END, "check_query"]:
    last_message = state["messages"][-1]
    return END if not last_message.tool_calls else "check_query"

# ==========================================
# 5. GRAPH COMPILATION & EXECUTION
# ==========================================
builder = StateGraph(AgentState)

builder.add_node("list_tables", list_tables)
builder.add_node("call_get_schema", call_get_schema)
builder.add_node("get_schema", get_schema_node)
builder.add_node("generate_query", generate_query)
builder.add_node("check_query", check_query)
builder.add_node("run_query", custom_run_query_node)  
builder.add_node("rewrite_question", rewrite_question)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_conditional_edges("run_query", grade_documents)
builder.add_edge("rewrite_question", "generate_query")
builder.add_edge("generate_answer", END)

agent = builder.compile()

# Terminal Execution Block
if __name__ == "__main__":
    # Optional: Save architecture map graph locally without breaking CLI execution
    try:
        agent.get_graph().draw_mermaid_png(output_file_path="sql_agent_graph.png")
        print("📊 Architecture flow map saved as 'sql_agent_graph.png'")
    except Exception:
        pass

    question = "ارزان ترین رب گوجه رو میخوام"
    final_state = None

    print(f"\nUser Question: {question}\nStreaming Execution steps:")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        final_state = step
        step["messages"][-1].pretty_print()

    # Formulate structural JSON payload for device frontend application interface
    final_device_payload = {
        "llm_response": final_state["messages"][-1].content if final_state else "",
        "products": final_state.get("raw_db_data", []) if final_state else []
    }

    print("\n Final Mobile Device Interface Payload JSON:")
    print(json.dumps(final_device_payload, ensure_ascii=False, indent=2))
