from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Any, Dict
import os
import uuid
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
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime

from config2 import load_config  

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

llm_openai = ChatOpenAI(model="gpt-4o", temperature=0)

# Global conversation history store
conversation_history: Dict[str, BaseChatMessageHistory] = {}

def get_conversation_history(session_id: str = "default") -> BaseChatMessageHistory:
    """Get or create conversation history for a session"""
    if session_id not in conversation_history:
        conversation_history[session_id] = ChatMessageHistory()
    return conversation_history[session_id]

system_template = """
You are a **PostgreSQL SQL AI assistant**. You generate SQL commands specifically for PostgreSQL database and EXECUTE them when requested.

IMPORTANT: Always use PostgreSQL syntax, not MySQL or other database systems:
- Use SERIAL or GENERATED ALWAYS AS IDENTITY for auto-incrementing columns (NOT AUTO_INCREMENT)
- Use TEXT or VARCHAR for strings
- Use BOOLEAN for boolean values (NOT TINYINT)
- Use TIMESTAMP for date/time (NOT DATETIME)

EXECUTION BEHAVIOR:
- When user asks to "create table", immediately use create_tables() tool
- When user says "yes" or "execute", use the appropriate tool for the last operation discussed
- When user asks to "use database", immediately use use_database() tool
- Be proactive and execute commands when clearly requested

Available helper tools:
 • create_database(db_name)
 • use_database(db_name) 
 • create_tables(sql_command) - Use PostgreSQL syntax only
 • drop_table(table_name)
 • list_tables()
 • describe_table(table_name)
 • insert_rows(sql_command, values_list)
 • select_rows(sql_command, values_list?)
 • update_rows(sql_command, values_list)
 • delete_rows(sql_command, values_list)

Examples of correct PostgreSQL syntax:
- CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(255), active BOOLEAN);
- CREATE TABLE orders (id GENERATED ALWAYS AS IDENTITY PRIMARY KEY, created_at TIMESTAMP);

Always output PostgreSQL-compatible SQL commands.
Execute commands immediately when users request table creation, database operations, etc.
"""

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{input}"),
])


class State(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_history: List[BaseMessage]  # Store persistent conversation history
    session_id: str  # Session identifier for tracking conversations


def add_to_history(state: State, message: BaseMessage) -> None:
    """Add a message to the conversation history"""
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append(message)
    
    # Also add to global history for persistence
    session_id = state.get("session_id", "default")
    history = get_conversation_history(session_id)
    history.add_message(message)


def get_context_summary(state: State, max_messages: int = 10) -> str:
    """Get a summary of recent conversation history for context"""
    history = state.get("conversation_history", [])
    if not history:
        return ""
    
    # Get last N messages for context
    recent_messages = history[-max_messages:]
    context_parts = []
    
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            context_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Skip tool calls in context, just show final responses
            if not msg.tool_calls:
                context_parts.append(f"Assistant: {msg.content}")
    
    return "\n".join(context_parts) if context_parts else ""


def _get_config(target_db: Optional[str] = None) -> dict:
    """Return connection credentials, overriding the database name if *target_db* is provided."""
    cfg = load_config().copy()
    cfg["dbname"] = target_db or os.getenv("ACTIVE_DB", cfg.get("dbname"))
    return cfg


def create_database(db_name: str) -> str:
    """
    Create a new database with the given name.
    :param db_name: Name of the database to create.
    :return: Success or error message.
    """
    conn = None
    try:
        # Connect without using context manager to avoid transaction block
        conn = psycopg2.connect(**_get_config("postgres"))
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if database already exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        
        if exists:
            return f"Database '{db_name}' already exists."
        
        # Create database
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        
        # Verify the database was created
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
    """
    Switch to the specified database.
    :param db_name: Name of the database to use.
    :return: Success or error message.
    """
    try:
        # First check if database exists by connecting to postgres and querying
        with psycopg2.connect(**_get_config("postgres")) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
                if not cur.fetchone():
                    return f"Database '{db_name}' does not exist. Please create it first."
        
        # Now try to connect to the target database
        with psycopg2.connect(**_get_config(db_name)):
            os.environ["ACTIVE_DB"] = db_name
            return f"Now using database '{db_name}'."
            
    except Exception as e:
        return f"Unable to switch database → {e}"


def create_tables(command: str) -> str:
    """
    Execute a SQL command to create tables.
    :param command: SQL command to create tables.
    :return: Success or error message.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(command)
        return "Table(s) created successfully."
    except Exception as e:
        return f"Unable to create table(s) → {e}"


def drop_table(table_name: str) -> str:
    """
    Drop a table with the given name.
    :param table_name: Name of the table to drop.
    :return: Success or error message.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE {};").format(sql.Identifier(table_name)))
        return f"Table '{table_name}' dropped successfully."
    except Exception as e:
        return f"Unable to drop table → {e}"


def list_tables() -> List[str]:
    """
    List all tables in the current database.
    :return: List of table names or an error message.
    """
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
    """
    Describe the structure of a table.
    :param table_name: Name of the table to describe.
    :return: List of column names and data types or an error message.
    """
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
    """
    Insert rows into a table.
    :param command: SQL command for inserting rows.
    :param values_list: List of values to insert (can be empty for complete SQL statements).
    :return: Success or error message.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            if values_list:
                # Use executemany for parameterized queries
                cur.executemany(command, values_list)
            else:
                # Use execute for complete SQL statements
                cur.execute(command)
            conn.commit()
        return "Rows inserted successfully."
    except Exception as e:
        return f"Unable to insert rows → {e}"


def select_rows(command: str, values: Optional[List[Any]] = None) -> List[tuple]:
    """
    Select rows from a table.
    :param command: SQL command for selecting rows.
    :param values: Optional list of values for parameterized queries.
    :return: List of rows or an error message.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(command, values or [])
            return cur.fetchall()
    except Exception as e:
        return [("error", str(e))]


def update_rows(command: str, values_list: List[tuple]) -> str:
    """
    Update rows in a table.
    :param command: SQL command for updating rows.
    :param values_list: List of values for parameterized queries.
    :return: Success or error message.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.executemany(command, values_list)
            conn.commit()
        return "Rows updated successfully."
    except Exception as e:
        return f"Unable to update rows → {e}"


def delete_rows(command: str, values_list: List[tuple]) -> str:
    """
    Delete rows from a table.
    :param command: SQL command for deleting rows.
    :param values_list: List of values for parameterized queries.
    :return: Success or error message.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.executemany(command, values_list)
            conn.commit()
        return "Rows deleted successfully."
    except Exception as e:
        return f"Unable to delete rows → {e}"

tools = [
    create_database,
    use_database,
    create_tables,
    drop_table,
    list_tables,
    describe_table,
    insert_rows,
    select_rows,
    update_rows,
    delete_rows,
]

llm_openai_with_tools = llm_openai.bind_tools(tools)


def sqlquerybot(state: State):
    # Get conversation context for better decision making
    context = get_context_summary(state)
    
    # Prepare the input with context if available
    current_input = state["messages"][-1].content if state["messages"] else ""
    
    # Add context to the prompt if available
    if context:
        enhanced_input = f"Previous conversation context:\n{context}\n\nCurrent request: {current_input}"
        # Replace the last message with the enhanced version
        messages = state["messages"][:-1] + [HumanMessage(content=enhanced_input)]
    else:
        messages = state["messages"]
    
    prompt = chat_prompt.format_prompt(input=messages).to_messages()
    response = llm_openai_with_tools.invoke(prompt)
    
    # Add the response to conversation history
    add_to_history(state, response)
    
    return {"messages": [response]}


def should_continue(state: State) -> str:
    """
    Determine if the conversation should continue or end.
    """
    last_message = state["messages"][-1] if state["messages"] else None
    
    # Check if the last message is a tool call (continue) or final response (end)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# Update your graph
graph = StateGraph(State)

graph.add_node("SqlQueryBot", sqlquerybot)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "SqlQueryBot")

# Add conditional edge from SqlQueryBot
graph.add_conditional_edges(
    "SqlQueryBot", 
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Tools go back to SqlQueryBot
graph.add_edge("tools", "SqlQueryBot")

agent = graph.compile()


# if __name__ == "__main__":
#     response = agent.invoke({
#         "messages": "Create database demo_db; then use database demo_db; create table products (id INT PRIMARY KEY, name TEXT, description TEXT)"
#     })
#     for m in response["messages"]:
#         m.pretty_print()


if __name__ == "__main__":
    print("🤖 SQL AI Assistant with History Management")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("Type 'new_session' to start a new conversation session")
    print("=" * 50)
    
    # Initialize session
    session_id = str(uuid.uuid4())[:8]  # Short session ID
    print(f"📝 Session ID: {session_id}")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                print("\n🤖 Assistant: Goodbye! Have a great day!")
                break
            
            # Check for new session command
            if user_input.lower() == 'new_session':
                session_id = str(uuid.uuid4())[:8]
                print(f"📝 New Session ID: {session_id}")
                print("🔄 Starting fresh conversation...")
                continue
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Process the user's message with session context
            print("\n🤖 Assistant: Processing your request...")
            
            # Initialize state with session information
            initial_state = {
                "messages": user_input,
                "session_id": session_id,
                "conversation_history": []
            }
            
            # Add user message to history
            user_message = HumanMessage(content=user_input)
            add_to_history(initial_state, user_message)
            
            response = agent.invoke(initial_state)
            
            # Display the response
            print("\n" + "=" * 50)
            for m in response["messages"]:
                m.pretty_print()
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different request.")