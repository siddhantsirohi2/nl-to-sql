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
from langgraph.prebuilt import ToolNode

from config2 import load_config

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

llm_openai = ChatOpenAI(model="gpt-4o", temperature=0)

# Global conversation history store
conversation_history_store: Dict[str, BaseChatMessageHistory] = {}

def get_conversation_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create conversation history for a session."""
    if session_id not in conversation_history_store:
        conversation_history_store[session_id] = ChatMessageHistory()
    return conversation_history_store[session_id]

# 1. ENHANCED SYSTEM PROMPT
system_template = """
You are a PostgreSQL SQL AI assistant. Your primary goal is to help users manage a PostgreSQL database by generating and executing SQL commands.

**Your Thought Process:**
1.  **Deconstruct the User's Request:** Break down the user's query into a logical sequence of steps. For example, "Create a database called 'sales', use it, and then add a 'customers' table" should be three distinct actions.
2.  **Select the Right Tool:** For each step, identify the single best tool from the available list.
3.  **Formulate the Tool Call:** Prepare the arguments for the selected tool. If the user provides a direct SQL command, pass it to the appropriate tool. If they ask in natural language, generate the correct PostgreSQL-compatible SQL first.
4.  **Execute and Respond:** Call the tool. If you encounter an error, explain it to the user and ask for clarification. If successful, confirm the action.

**Important Rules:**
- **PostgreSQL Syntax Only:** Always use PostgreSQL syntax (e.g., `SERIAL` or `GENERATED ALWAYS AS IDENTITY`, `TEXT`, `BOOLEAN`, `TIMESTAMP`). Do NOT use MySQL syntax like `AUTO_INCREMENT`.
- **Proactive Execution:** When a user's intent is clear (e.g., "create this table," "add these rows"), execute the command immediately without asking for confirmation. If the request is ambiguous, ask clarifying questions.
"""

# 2. IMPROVED STATE MANAGEMENT WITH MessagesPlaceholder
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="history"),  # Use a placeholder for history
    HumanMessagePromptTemplate.from_template("{input}"),
])


class State(TypedDict):
    messages: Annotated[list, add_messages]
    history: List[BaseMessage]
    session_id: str


def _get_config(target_db: Optional[str] = None) -> dict:
    """Return connection credentials, overriding the database name if *target_db* is provided."""
    cfg = load_config().copy()
    cfg["dbname"] = target_db or os.getenv("ACTIVE_DB", cfg.get("dbname"))
    return cfg


def create_database(db_name: str) -> str:
    """Creates a new PostgreSQL database."""
    try:
        conn = psycopg2.connect(**_get_config("postgres"))
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone():
                return f"Database '{db_name}' already exists."
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        return f"Database '{db_name}' created successfully."
    except Exception as e:
        return f"Unable to create database → {e}"
    finally:
        if conn:
            conn.close()


def use_database(db_name: str) -> str:
    """Switches the active database connection for subsequent operations."""
    try:
        with psycopg2.connect(**_get_config("postgres")) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
            if not cur.fetchone():
                return f"Database '{db_name}' does not exist. Please create it first."
        with psycopg2.connect(**_get_config(db_name)):
            os.environ["ACTIVE_DB"] = db_name
            return f"Now using database '{db_name}'."
    except Exception as e:
        return f"Unable to switch database → {e}"


def create_tables(command: str) -> str:
    """Executes a SQL command, typically CREATE TABLE, in the active database."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(command)
        return "Table(s) created successfully."
    except Exception as e:
        return f"Unable to create table(s) → {e}"


def drop_table(table_name: str) -> str:
    """Drops a table with the given name from the active database."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(table_name)))
        return f"Table '{table_name}' dropped successfully."
    except Exception as e:
        return f"Unable to drop table → {e}"


def list_tables() -> List[str]:
    """Lists all tables in the 'public' schema of the current database."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        return [f"Unable to list tables → {e}"]


def describe_table(table_name: str) -> List[tuple]:
    """Describes the structure (columns and data types) of a given table."""
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position;", (table_name,)
            )
            return cur.fetchall()
    except Exception as e:
        return [("error", str(e))]

# 3. ENHANCED TOOL DESCRIPTIONS (DOCSTRINGS)
def insert_rows(command: str, values_list: Optional[List[tuple]] = None) -> str:
    """
    Executes an INSERT statement to add one or more rows to a table.

    Use cases:
    1. Full SQL Command: Provide the complete INSERT statement in `command` and an empty `values_list`.
       - Ex: `command="INSERT INTO users (name) VALUES ('Alice');"`, `values_list=[]`
    2. Parameterized Query: Provide a template SQL in `command` with `%s` placeholders and a list of tuples in `values_list`.
       - Ex: `command="INSERT INTO users (name, email) VALUES (%s, %s);"`, `values_list=[('Bob', 'bob@a.com'), ('Charlie', 'charlie@b.com')]`

    :param command: The SQL INSERT statement or a parameterized template.
    :param values_list: A list of tuples, where each tuple is a row to insert. Default is None.
    :return: A string indicating success or failure.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            if values_list:
                cur.executemany(command, values_list)
            else:
                cur.execute(command)
            conn.commit()
        return "Rows inserted successfully."
    except Exception as e:
        return f"Unable to insert rows → {e}"


def select_rows(command: str, values: Optional[List[Any]] = None) -> List[tuple]:
    """
    Executes a SELECT statement to query rows from a table.

    :param command: The SQL SELECT statement, which can include `%s` placeholders.
    :param values: An optional list of values for parameter substitution in the command.
    :return: A list of result rows or an error message.
    """
    try:
        with psycopg2.connect(**_get_config()) as conn, conn.cursor() as cur:
            cur.execute(command, values or [])
            return cur.fetchall()
    except Exception as e:
        return [("error", str(e))]


def update_rows(command: str, values_list: List[tuple]) -> str:
    """
    Executes an UPDATE statement on one or more rows.

    :param command: The parameterized SQL UPDATE statement (e.g., "UPDATE users SET email = %s WHERE id = %s;").
    :param values_list: A list of tuples containing the values for substitution.
    :return: A string indicating success or failure.
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
    Executes a DELETE statement on one or more rows.

    :param command: The parameterized SQL DELETE statement (e.g., "DELETE FROM users WHERE id = %s;").
    :param values_list: A list of tuples containing the values for substitution.
    :return: A string indicating success or failure.
    """
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

llm_with_tools = llm_openai.bind_tools(tools)

# 3. SIMPLIFIED AGENT LOGIC
def sqlquerybot(state: State):
    """The main agent node."""
    # Get the session's history and the current user input
    history = state["history"]
    input_message = state["messages"][-1]

    # Format the prompt properly with history and current input
    prompt_messages = chat_prompt.format_messages(
        history=history,
        input=input_message.content
    )

    # Invoke the LLM with the formatted messages
    response = llm_with_tools.invoke(prompt_messages)

    return {"messages": [response]}


def should_continue(state: State) -> str:
    """Determines the next step: call tools or end the turn."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# Define the graph
graph = StateGraph(State)
graph.add_node("SqlQueryBot", sqlquerybot)
graph.add_node("tools", ToolNode(tools))

# Define the edges
graph.add_edge(START, "SqlQueryBot")
graph.add_conditional_edges("SqlQueryBot", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "SqlQueryBot")

agent = graph.compile()

# 4. REFINED MAIN LOOP
if __name__ == "__main__":
    print("🤖 SQL AI Assistant with History Management")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("Type 'new_session' to start a new conversation session")
    print("=" * 50)

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    print(f"📝 Session ID: {session_id}")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                print("\n🤖 Assistant: Goodbye! Have a great day!")
                break

            if user_input.lower() == 'new_session':
                session_id = f"session_{uuid.uuid4().hex[:8]}"
                print(f"📝 New Session ID: {session_id}")
                print("🔄 Starting fresh conversation...")
                continue

            if not user_input:
                continue

            print("\n🤖 Assistant: Processing your request...")
            
            # Get the persistent history for the current session
            history = get_conversation_history(session_id)

            # Define the input for the agent
            agent_input = {
                "messages": [HumanMessage(content=user_input)],
                "history": history.messages,
                "session_id": session_id,
            }

            # Invoke the agent
            response = agent.invoke(agent_input)
            
            # Update the persistent history with both the user's message and the AI's final response
            history.add_user_message(user_input)
            history.add_ai_message(response["messages"][-1])

            # Display the final response
            print("\n" + "=" * 50)
            response["messages"][-1].pretty_print()
            print("=" * 50)

        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")