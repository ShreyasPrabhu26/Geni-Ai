import os
from mem0 import Memory
from google import genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

QUADRANT_HOST = "localhost"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": GOOGLE_API_KEY,
            "model": "gemini-embedding-001",
            "output_dimensionality": 1536
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": GOOGLE_API_KEY,
            "model": "gemini-2.5-flash"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        },
    },
}

mem_client = Memory.from_config(config)
client = genai.Client()
chat = client.chats.create(model="gemini-2.5-flash")

# Persistent message list
messages = []


def chatWithAi(message):
    messages.append({"role": "user", "content": message})

    # Send message to AI
    response = chat.send_message(message)
    reply = response.text
    print("🤖:", reply)
    messages.append({"role": "assistant", "content": reply})

    # Store full chat context
    mem_client.add(messages, user_id="Shreyas Prabhu")


while True:
    user_input = input("👨🏻‍💻:")
    if user_input.lower() in ["exit", "quit"]:
        break
    chatWithAi(user_input)
