from pathlib import Path
from dotenv import load_dotenv

# Document loader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# Embedding store
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

path_path = Path(__file__).parent / "node-dev.pdf"

loader = PyPDFLoader(file_path=path_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size: the maximum number of characters in each chunk
    chunk_size=1000,
    # chunk_overlap: the number of characters to overlap between chunks
    # this allows the model to capture context between chunks
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(docs)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# ==================================
# Injection Phase.
# vetor_store = QdrantVectorStore.from_documents(
#     documents=[],
#     collection_name="genai-pdf",
#     url="http://localhost:6333",
#     embedding=embedder,
# )

# vetor_store.add_documents(split_docs)

# print("Documents added to the vector store.")
# ==================================

# Retrival Phase.

retriever = QdrantVectorStore.from_existing_collection(
    collection_name="genai-pdf",
    url="http://localhost:6333",
    embedding=embedder,
)

relavent_chunks = retriever.similarity_search(
    query="What is Node.js?",
)

# print("Relavent Chunks",relavent_chunks)

# summarization phase.
import json

SYSTEM_PROMPT = """
You are a helpful AI assistant who responds based on the relevant chunks of information provided.
Context: {relevant_chunks}
"""

# Format the relevant chunks as text for the prompt
chunks_text = "\n\n".join([chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in relavent_chunks])

prompt = SYSTEM_PROMPT.format(relevant_chunks=chunks_text)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    ("system", prompt),
    ("human", "Answer the user's question based on the context above.")
]

ai_msg = llm.invoke(messages)
print("AI Response:", ai_msg.content)