from graph import graph, create_Chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "1"}}


def init():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_Chat_graph(checkpointer=checkpointer)

        while True:
            user_input = input(">: ")
            for event in graph_with_mongo.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()


init()
