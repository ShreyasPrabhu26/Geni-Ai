from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

def run_retrieval_phase(collection_name, qdrant_url, embedding_model, query):
    embedder = GoogleGenerativeAIEmbeddings(
        model=embedding_model
    )
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        url=qdrant_url,
        embedding=embedder,
    )
    relavent_chunks = retriever.similarity_search(
        query=query,
    )
    print("Relavent Chunks", relavent_chunks)
    return relavent_chunks
