from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Any, Dict
import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from datetime import datetime
import json
import re

from config2 import load_config  

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

llm_openai = ChatOpenAI(model="gpt-4o", temperature=0)

# Global chat history store
chat_histories: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

# Contextualization Agent
contextualization_system_template = """
You are a **Query Contextualization Agent** for a PostgreSQL SQL assistant. Your job is to analyze the user's current query in the context of recent conversation history and enhance it with relevant context.

**Your Tasks:**
1. Analyze the conversation history to understand:
   - What databases and tables have been created/used
   - What operations were performed recently
   - Any references to "the table", "that database", pronouns, or implicit references
   
2. Enhance the user query by:
   - Adding explicit table/database names when the user uses pronouns or implicit references
   - Providing context about what the user is likely referring to
   - Clarifying ambiguous requests based on recent operations
   
3. Output a JSON object with:
   - "contextualized_query": Enhanced version of the user's query with explicit context
   - "relevant_context": Key context information that influenced your decision
   - "confidence": Your confidence level (high/medium/low) in the contextualization

**Examples:**
User says: "Now add some sample data to it"
Recent history shows: Created table 'users' with columns id, name, email
Output: {{
  "contextualized_query": "Now add some sample data to the 'users' table that we just created",
  "relevant_context": "Recently created 'users' table with columns id, name, email",
  "confidence": "high"
}}

User says: "Show me the structure"
Recent history shows: Working with 'products' table
Output: {{
  "contextualized_query": "Show me the structure of the 'products' table",
  "relevant_context": "Currently working with 'products' table",
  "confidence": "high"
}}

If the query is already clear and explicit, return it unchanged but still provide the context analysis.
"""

contextualization_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(contextualization_system_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("Current user query: {query}")
])

# Main SQL Agent
sql_system_template = """
You are a **PostgreSQL SQL AI assistant**. You generate SQL commands specifically for PostgreSQL database.

IMPORTANT: Always use PostgreSQL syntax, not MySQL or other database systems:
- Use SERIAL or GENERATED ALWAYS AS IDENTITY for auto-incrementing columns (NOT AUTO_INCREMENT)
- Use TEXT or VARCHAR for strings
- Use BOOLEAN for boolean values (NOT TINYINT)
- Use TIMESTAMP for date/time (NOT DATETIME)
- Use %s for parameters, NOT $1, $2, $3 syntax

The user's query has been contextualized based on conversation history. Use this enhanced understanding to provide better responses.

**Context Information:**
{context_info}

Available helper tools:
 • create_database(db_name)
 • use_database(db_name) 
 • create_tables(sql_command) - Use PostgreSQL syntax only
 • drop_table(table_name)
 • list_tables()
 • describe_table(table_name) - Use when you need to know table structure
 • insert_rows(sql_command, values_list) - Insert data into tables
 • select_rows(sql_command, values_list?)
 • update_rows(sql_command, values_list) - Update data directly
 • delete_rows(sql_command, values_list)

EXECUTION RULES:
- For INSERT operations: Use insert_rows() directly when you have the data
- For UPDATE operations: Use update_rows() directly when you have the condition and new values
- For SELECT operations: Use select_rows() directly
- For DELETE operations: Use delete_rows() directly
- Only use list_tables() or describe_table() when explicitly asked for table information
- Always use %s for parameters in SQL commands, never $1, $2, $3

Examples:
- UPDATE employees SET salary = %s WHERE id = %s
- INSERT INTO employees (name, email) VALUES (%s, %s)
- DELETE FROM employees WHERE id = %s

Always output PostgreSQL-compatible SQL commands.
Execute the user's request directly and efficiently.
"""

sql_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(sql_system_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

class State(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    contextualized_query: str
    context_info: str
    session_id: str

def _get_config(target_db: Optional[str] = None) -> dict:
    """Return connection credentials, overriding the database name if *target_db* is provided."""
    cfg = load_config().copy()
    cfg["dbname"] = target_db or os.getenv("ACTIVE_DB", cfg.get("dbname"))
    return cfg

# Database tools (same as before but with better error handling)
def create_database(db_name: str) -> str:
    """Create a new database with the given name."""
    conn = None
    try:
        conn = psycopg2.connect(**_get_config("postgres"))
        conn.autocommit = True
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        
        if exists:
            return f"Database '{db_name}' already exists."
        
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        if not cur.fetchone():
            return f"Failed to create database '{db_name}'."
        
        cur.close()
        return f"Database '{db_name}' created successfully."
        
    except Exception as e:
        return f"Unable to create database → {e}"
    finally:
        if conn:
            conn.close()

def use_database(db_name: str) -> str:
    """Switch to the specified database."""
    try:
        with psycopg2.connect(**_get_config("postgres")) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
                if not cur.fetchone():
                    return f"Database '{db_name}' does not exist. Please create it first."
        
        with psycopg2.connect(**_get_config(db_name)):
            os.environ["ACTIVE_DB"] = db_name
            return f"Now using database '{db_name}'."
            
    except Exception as e:
        return f"Unable to switch database → {e}"

def create_tables(command: str) -> str:
    """Execute a SQL command to create tables."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(command)
        return "Table(s) created successfully."
    except Exception as e:
        return f"Unable to create table(s) → {e}"

def drop_table(table_name: str) -> str:
    """Drop a table with the given name."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE {};").format(sql.Identifier(table_name)))
        return f"Table '{table_name}' dropped successfully."
    except Exception as e:
        return f"Unable to drop table → {e}"

def list_tables() -> List[str]:
    """List all tables in the current database."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' ORDER BY table_name;
                """
            )
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        return [f"Unable to list tables → {e}"]

def describe_table(table_name: str) -> List[tuple]:
    """Describe the structure of a table."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT column_name, data_type FROM information_schema.columns
                    WHERE table_name = %s ORDER BY ordinal_position;
                    """
                ),
                (table_name,),
            )
            return cur.fetchall()
    except Exception as e:
        return [("error", str(e))]

def insert_rows(command: str, values_list: List[tuple]) -> str:
    """Insert rows into a table."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            if values_list:
                # Convert $1, $2, $3 syntax to %s for psycopg2
                if '$' in command:
                    import re
                    command = re.sub(r'\$\d+', '%s', command)
                
                # Convert list of lists to tuples if needed
                if values_list and isinstance(values_list[0], list):
                    values_list = [tuple(row) for row in values_list]
                
                cur.executemany(command, values_list)
            else:
                cur.execute(command)
            conn.commit()
        return "Rows inserted successfully."
    except Exception as e:
        return f"Unable to insert rows → {e}"

def select_rows(command: str, values: Optional[List[Any]] = None) -> List[tuple]:
    """Select rows from a table."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(command, values or [])
            return cur.fetchall()
    except Exception as e:
        return [("error", str(e))]

def update_rows(command: str, values_list: List[tuple]) -> str:
    """Update rows in a table."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.executemany(command, values_list)
            conn.commit()
        return "Rows updated successfully."
    except Exception as e:
        return f"Unable to update rows → {e}"

def delete_rows(command: str, values_list: List[tuple]) -> str:
    """Delete rows from a table."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.executemany(command, values_list)
            conn.commit()
        return "Rows deleted successfully."
    except Exception as e:
        return f"Unable to delete rows → {e}"

tools = [
    create_database, use_database, create_tables, drop_table,
    list_tables, describe_table, insert_rows, select_rows,
    update_rows, delete_rows,
]

llm_openai_with_tools = llm_openai.bind_tools(tools)

def contextualizer_agent(state: State):
    """Agent that contextualizes user queries based on conversation history"""
    session_id = state.get("session_id", "default")
    history = get_session_history(session_id)
    
    # Get recent messages for context (last 10 messages)
    recent_messages = history.messages[-10:] if len(history.messages) > 10 else history.messages
    
    try:
        # Create contextualization chain with message history
        contextualization_chain = contextualization_prompt | llm_openai
        
        response = contextualization_chain.invoke({
            "query": state["original_query"],
            "chat_history": recent_messages
        })
        
        # Parse the JSON response
        try:
            context_data = json.loads(response.content)
            contextualized_query = context_data.get("contextualized_query", state["original_query"])
            context_info = context_data.get("relevant_context", "No additional context")
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            contextualized_query = state["original_query"]
            context_info = "Context analysis failed, using original query"
            
    except Exception as e:
        # Fallback in case of any error
        contextualized_query = state["original_query"]
        context_info = f"Contextualization error: {e}"
    
    return {
        "contextualized_query": contextualized_query,
        "context_info": context_info
    }

def sqlquerybot(state: State):
    """Main SQL agent that processes contextualized queries"""
    session_id = state.get("session_id", "default")
    history = get_session_history(session_id)
    
    # Get recent messages for context
    recent_messages = history.messages[-5:] if len(history.messages) > 5 else history.messages
    
    # Use the contextualized query
    query_to_process = state.get("contextualized_query", state["original_query"])
    context_info = state.get("context_info", "No context available")
    
    prompt_messages = sql_prompt.format_prompt(
        input=query_to_process,
        context_info=context_info,
        chat_history=recent_messages
    ).to_messages()
    
    response = llm_openai_with_tools.invoke(prompt_messages)
    
    # Add messages to history
    history.add_user_message(state["original_query"])
    history.add_ai_message(response.content)
    
    return {"messages": [response]}

def should_continue(state: State) -> str:
    """Determine if the conversation should continue or end."""
    last_message = state["messages"][-1] if state["messages"] else None
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# Create the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("contextualizer", contextualizer_agent)
graph.add_node("sqlquerybot", sqlquerybot)
graph.add_node("tools", ToolNode(tools))

# Add edges - simple linear flow
graph.add_edge(START, "contextualizer")
graph.add_edge("contextualizer", "sqlquerybot")

# Add conditional edge from SqlQueryBot
graph.add_conditional_edges(
    "sqlquerybot", 
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# End after tools - no loop back
graph.add_edge("tools", END)

agent = graph.compile()

def process_query(user_input: str, session_id: str = "default") -> dict:
    """Process a user query with contextualization"""
    return agent.invoke(
        {
            "original_query": user_input,
            "messages": [],
            "session_id": session_id
        },
        config={"recursion_limit": 25}
    )

if __name__ == "__main__":
    print("🤖 SQL AI Assistant with Smart Context")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("=" * 60)
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                print("\n🤖 Assistant: Goodbye! Have a great day!")
                break
                
            if user_input.lower() == 'clear':
                if session_id in chat_histories:
                    chat_histories[session_id].clear()
                print("\n🤖 Assistant: Conversation history cleared!")
                continue
            
            if not user_input:
                continue
            
            print("\n🤖 Assistant: Analyzing context and processing your request...")
            
            # Process the query with contextualization
            response = process_query(user_input, session_id)
            
            # Display the response
            print("\n" + "=" * 60)
            
            # Show contextualization info if available
            if "context_info" in response and response["context_info"] != "No context available":
                print(f"🔍 Context: {response['context_info']}")
                print("-" * 40)
            
            # Show messages
            for m in response["messages"]:
                m.pretty_print()
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different request.")