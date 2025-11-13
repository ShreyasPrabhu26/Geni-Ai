from dotenv import load_dotenv
from voice.recognizer import listen_and_recognize
from langgraph.checkpoint.mongodb import MongoDBSaver  
from graph.graph import create_chat_graph
from text_to_voice import speak_text

load_dotenv()

MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "1"}}

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer: 
        graph = create_chat_graph(checkpointer=checkpointer) 
        while True:
            print("\n\nStart talking:")
            success, result = listen_and_recognize()
            if success:
                print("User said:", result)
                ai_response = None
                for event in graph.stream({"messages":[{"role":"user","content":result}]},config,stream_mode="values"):
                    if "messages" in event:
                        last_message = event["messages"][-1]
                        last_message.pretty_print()
                        # Capture AI response (not user message)
                        if hasattr(last_message, 'content') and last_message.type != "human":
                            ai_response = last_message.content
                
                # Speak the AI response
                if ai_response:
                    print("\nSpeaking AI response...")
                    # Extract text from response (handle list of dicts or plain string)
                    if isinstance(ai_response, list):
                        text_to_speak = " ".join([item.get('text', '') for item in ai_response if isinstance(item, dict) and 'text' in item])
                    else:
                        text_to_speak = str(ai_response)
                    
                    if text_to_speak.strip():
                        speak_text(text_to_speak)
            else:
                print(result)

if __name__ == "__main__":
    main()
