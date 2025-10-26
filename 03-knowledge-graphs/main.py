import os
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory

# Load environment variables
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Local configurations ---
QUADRANT_HOST = "localhost"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"

# --- Memory configuration ---
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

# --- Initialize memory and LLM clients ---
mem_client = Memory.from_config(config)

client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Persistent chat memory
messages = []


def chatWithAi(message):
    # Retrieve contextual memories
    mem_result = mem_client.search(query=message, user_id="Shreyas Prabhu")
    memories = [res["memory"] for res in mem_result.get("results", [])]

    if memories:
        print("🧠 Retrieved from memory:", memories)

    # --- Construct dynamic system prompt ---
    SYSTEM_PROMPT = f"""
    You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
    systematically analyze input content, extract structured knowledge, and maintain an
    optimized memory store. Your primary function is information distillation
    and knowledge preservation with contextual awareness.

    Tone: Professional analytical, precision-focused, with clear uncertainty signaling.

    Memory and Score:
    {memories}
    """

    # --- Build full chat messages ---
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,  # include prior conversation
        {"role": "user", "content": message},
    ]

    # --- Send to Gemini (via OpenAI SDK) ---
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=chat_messages
    )

    reply = response.choices[0].message.content
    print("🤖:", reply)

    # Append to persistent messages for context continuity
    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": reply})

    # --- Save conversation to memory ---
    mem_client.add(messages, user_id="Shreyas Prabhu")


if __name__ == "__main__":
    print("🤖 Gemini Memory Chat (Ctrl+C or type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("👨🏻‍💻: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            chatWithAi(user_input)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("⚠️ Error:", e)
