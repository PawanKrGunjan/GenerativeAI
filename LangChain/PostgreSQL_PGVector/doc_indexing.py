from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Load PDF
loader = PyPDFLoader("pawankrgunjan.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = splitter.split_documents(docs)

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# PostgreSQL connection
CONNECTION = (
    f"postgresql+psycopg://postgres:"
    f"{os.getenv('POSTGRE_PASSWORD')}"
    f"@localhost:5433/rag_test"
)

# Store in PGVector
pgvector = PGVector.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="docs",   # ✅ Use ONE name everywhere
    connection=CONNECTION,
    use_jsonb=True
)

print("Documents indexed successfully!")