from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
# Reducers
from typing import Annotated
from langgraph.graph.message import add_messages
import psycopg2
from config import load_config
from typing import List
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()

llm_google = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

system_template = """
                You are an SQL AI assistant. You can create SQL queries
                which needed in python to run the query. For example,
                Human Message: insert rows in table vendor with vendor_id 1 and vendor_name TCS
                AI message: 
                command = "insert into vendor (vendor_id, vendor_name) values (%s, %s)"
                value_list = [(1,"TCS")] 
                Please reply appropriately for small talks and greetings.
                """
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

print(chat_prompt.input_variables)


def create_tables(command: str) -> str:
    """
    Execute Create table SQL queries by passing
    it as str
    :param command str: SQL query
    :return: str
    """

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                # execute create table query
                cur.execute(command)
        return "Create table successfully"
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
        return "Unable to create table"
    finally:
        if conn:
            cur.close()
            conn.close()


def insert_rows(command, values_list):
    """
    Execute insert row SQL queries by passing
    sql query and values
    :param command: str
    :param values_list: list
    :return: str
    """
    try:
        config = load_config()
        with  psycopg2.connect(**config) as conn:
            with  conn.cursor() as cur:
                # execute the INSERT statement
                cur.executemany(command, values_list)
            # commit the changes to the database
            conn.commit()
            return "insert data successfully"
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return "Unable to insert data"
    finally:
        if conn:
            cur.close()
            conn.close()

llm_google_with_tools=llm_google.bind_tools([create_tables,insert_rows])

tools=[create_tables,insert_rows]
def sqlquerybot(state: State):
    prompt = chat_prompt.format_prompt(input=state['messages']).to_messages()
    return {"messages": [llm_google_with_tools.invoke(prompt)]}


graph = StateGraph(State)
## node
graph.add_node("SqlQueryBot", sqlquerybot)
graph.add_node("tools",ToolNode(tools))
## Edges

graph.add_edge(START, "SqlQueryBot")
graph.add_conditional_edges(
    "SqlQueryBot",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
graph.add_edge("tools",END)


graph_builder = graph.compile()

result = graph_builder.invoke({'messages': "Create table products with product id, product name and product description"})

for message in result["messages"]:
    message.pretty_print()

