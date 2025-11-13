from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def run_command(cmd:str):
    """
    Takes a command line prompt and executes it on users machine and returns the output of the command
    """
    return os.system(command=cmd)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm_with_tools = llm.bind_tools([run_command])

def chatbot(state: State):
    system_prompt = SystemMessage(content="""You are an ai coding assistanr who takes input from user and based on the avaiable tools you choose correct tool to execute the commands.

     you can even execute commands and help teh user with the output of the command.
    Always make sure to keep the generated files in assistant_generated / folder you can create one if there is not already.
    """)
    message = llm_with_tools.invoke([system_prompt]+state["messages"])
    return {"messages": [message]}
    
tool_node = ToolNode(tools=[run_command])

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