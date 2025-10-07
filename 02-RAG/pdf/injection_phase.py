from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

def run_injection_phase(pdf_path, collection_name, qdrant_url, embedding_model, chunk_size, chunk_overlap):
    load_dotenv()
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(docs)
    embedder = GoogleGenerativeAIEmbeddings(
        model=embedding_model
    )
    QdrantVectorStore.from_documents(
        documents=split_docs,
        collection_name=collection_name,
        url=qdrant_url,
        embedding=embedder,
    )
    print("Documents added to the vector store.")
    return split_docs
