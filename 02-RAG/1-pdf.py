from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

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

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
vector = embeddings.embed_query("hello, world!")
print(vector)