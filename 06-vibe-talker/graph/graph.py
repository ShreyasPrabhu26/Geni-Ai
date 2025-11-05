from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv


load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def chatbot(state: State):
    message = llm.invoke(state["messages"])
    return {"messages": [message]}

tool_node = ToolNode(tools=[])

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",tool_node)

graph_builder.add_edge(START,"chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)