from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

import streamlit as st

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing")


################## LLM ##################

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="qwen/qwen3.8-27b",
    temperature=0,
)


################## State ##################

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


################## Chat Node ##################

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


################## PostgreSQL Connection ##################

# Neon PostgreSQL connection
# autocommit and row_factory are configured by PostgresSaver.from_conn_string

checkpointer_context = PostgresSaver.from_conn_string(DATABASE_URL)

checkpointer = checkpointer_context.__enter__()

# Create LangGraph checkpoint tables if they do not exist
checkpointer.setup()


################## Create Graph ##################

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")

graph.add_edge("chat", END)

# Compile graph with PostgreSQL persistence
chatbot = graph.compile(checkpointer=checkpointer)


################## Retrieve Threads ##################

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config["configurable"]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)