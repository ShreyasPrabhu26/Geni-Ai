from pathlib import Path
from injection_phase import run_injection_phase
from retrieval_phase import run_retrieval_phase
from summarization_phase import run_summarization_phase


# Constants (sorted by execution phase)

# Injection Phase
PDF_PATH = Path(__file__).parent / "node-dev.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "genai-pdf"
QDRANT_URL = "http://localhost:6333"

# Retrieval Phase
QUERY = "how to install modules in nodejs?"

# Summarization Phase
SYSTEM_PROMPT = """
You are a helpful AI assistant who responds based on the relevant chunks of information provided.\nContext: {relevant_chunks}
"""
LLM_MODEL = "gemini-2.5-flash"

if __name__ == "__main__":
    # Injection Phase
    run_injection_phase(
        pdf_path=PDF_PATH,
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Retrieval Phase
    relavent_chunks = run_retrieval_phase(
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
        embedding_model=EMBEDDING_MODEL,
        query=QUERY,
    )

    # Summarization Phase
    run_summarization_phase(
        relavent_chunks=relavent_chunks,
        system_prompt=SYSTEM_PROMPT,
        llm_model=LLM_MODEL,
        user_question=QUERY,
    )