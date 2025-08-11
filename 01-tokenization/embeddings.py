from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

input_text = "Hello, how are you?"
response = client.embeddings.create(
    input=input_text,
    model="text-embedding-3-small"
)

print(f"Embedding for input text: {input_text}")
print(response.data[0].embedding)