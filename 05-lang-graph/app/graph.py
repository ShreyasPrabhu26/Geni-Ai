from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    messages = state.get("messages")
    response = llm.invoke(messages)
    return {
        "messages": [response]
    }


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# Creates graph without memory
graph = graph_builder.compile()

# Creates graph with memory


def create_Chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
