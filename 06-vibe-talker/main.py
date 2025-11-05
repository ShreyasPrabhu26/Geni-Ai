from dotenv import load_dotenv
from voice.recognizer import listen_and_recognize
from langgraph.checkpoint.mongodb import MongoDBSaver  
from graph.graph import create_chat_graph

load_dotenv()

MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "1"}}

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer: 
        graph = create_chat_graph(checkpointer=checkpointer) 
        print("\n\nStart talking:")
        success, result = listen_and_recognize()
        if success:
            print("Bot said:", result)
            for event in graph.stream({"messages":[{"role":"user","content":result}]},config,stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        else:
            print(result)

if __name__ == "__main__":
    main()
